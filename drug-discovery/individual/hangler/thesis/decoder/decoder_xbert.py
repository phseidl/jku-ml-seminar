import torch
import pytorch_lightning as pl
from torch import nn
from xbert import BertConfig, BertForMaskedLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import create_scheduler


class XbertDecoder(pl.LightningModule):
    def __init__(
        self,
        config,
        pad_idx=0,
        embed_dim=None,
        use_linear=False,
        vocab=None
    ):
        """
        Args:
            config_path: path to the BERT config JSON.
            pad_idx: token id for <pad>.
            lr: learning rate.
            weight_decay: weight decay for optimizer.
            embed_dim: dimension of your stored embeddings (e.g., 768).
            use_linear: whether to apply a linear dimensional adaptation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True

        self.pad_idx = pad_idx
        self.use_linear = use_linear
        self.training_step_outputs = []
        self.config = config
        self.vocab = vocab

        # 1) Load the BERT config and create the model
        self.bert_config = BertConfig.from_json_file(config['bert_config_decoder'])

        self.bert_config.add_cross_attention = True
        self.bert_config.is_decoder = True
        self.model = BertForMaskedLM(config=self.bert_config)

        # 2) Optional linear layers if stored embeddings != BERT hidden size
        if use_linear:
            # E.g., going from 768 -> 64 -> 768, if your final_dim == 768
            self.output_dim = 64
            final_dim = self.bert_config.encoder_width  # e.g. 768
            self.encode_prefix = nn.Linear(final_dim, self.output_dim)
            self.decode_prefix = nn.Linear(self.output_dim, final_dim)
        
        self.warmup_steps = config['schedular']['warmup_epochs']

        # 3) Loss function (sum reduction can be OK, but often "mean" is used)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='sum')

    def forward(self, context_embeddings, input_ids, attention_mask=None):
        """
        context_embeddings: shape [batch_size, embed_dim].
        input_ids: shape [batch_size, seq_len].
        attention_mask: shape [batch_size, seq_len], if needed.
        Returns:
            logits: shape [batch_size, seq_len, vocab_size].
        """

        # If we have prefix layers, adapt dimension (e.g., 768 -> 64 -> 768).
        if self.use_linear and self.encode_prefix and self.decode_prefix:
            context_embeddings = self.decode_prefix(self.encode_prefix(context_embeddings))
        # shape is now [batch_size, final_dim], e.g., 768

        # Expand context to [batch_size, seq_len, final_dim]
        batch_size, seq_len = input_ids.shape
        encoder_hidden_states = context_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        # Forward pass through BERT. Return logits for next-token prediction.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,            # mask for the input tokens
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            return_dict=True,
            is_decoder=True
        )
        #print("outputs forward: ", outputs)
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
        return logits

    def training_step(self, batch, batch_idx):
        """
        batch: (context_embeddings, input_ids, target_ids)
        Typically your DataModule will create input_ids/target_ids as "teacher-forced" pairs
        (e.g. input_ids = tokens[:-1], target_ids = tokens[1:]).
        """
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        # batch has format: emb_tensor, input_ids, target_ids
        context_embeddings, input_ids, target_ids = batch
        attention_mask = (input_ids != self.pad_idx).long()

        # 1) Forward pass => logits
        logits = self(context_embeddings, input_ids, attention_mask=attention_mask)
        # shape [batch_size, seq_len, vocab_size]

        # 2) Compute cross-entropy
        # target_ids shape: [batch_size, seq_len]
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        # 2) Manual backward if not using standard automatic_optimization
        #if loss != 0:
        #    self.manual_backward(loss)
        #    nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        #    optimizer.step()
        #else:
        #    print('Loss is zero... check your data/inputs?')

        # 3) Logging
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss_mlm', loss, prog_bar=True)

        # 3) Compute accuracy just for logging
        preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        mask = (target_ids != self.pad_idx)
        correct = ((preds == target_ids) & mask).sum().float()
        total = mask.sum().float().clamp_min(1.0)
        acc = correct / total

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # step_size = 100
        # warmup_iterations = self.warmup_steps * step_size
        # if self.current_epoch > 0 and batch_idx == 0:
        #     scheduler.step(self.current_epoch + self.warmup_steps)
        # else:
        #     if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
        #         scheduler.step(batch_idx // step_size)
        # self.training_step_outputs.append(torch.tensor([loss, ]))
        
        return loss

    def on_train_epoch_end(self):
        # 1) Log the teacher-forced loss as before
        # tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        # if self.global_rank == 0:
        #     print(f'\n mean loss: {tmp[0]:.4f}')
        # self.training_step_outputs.clear()

        # 2) Check free-decoding accuracy on a small training subset
        #    (We do this on rank zero only to avoid duplication in multi-GPU)
        if self.global_rank == 0:
            train_loader = self.trainer.datamodule.train_dataloader()
            # fetch a single mini-batch
            try:
                batch = next(iter(train_loader))
            except StopIteration:
                # if the dataloader is empty, skip
                return

            # Move batch to the same device as the model
            context_embeddings, input_ids, target_ids = [
                x.to(self.device) for x in batch
            ]

            # Optionally limit to a small subset of the batch, e.g. first 8 samples
            subset_size = min(8, context_embeddings.size(0))
            context_embeddings = context_embeddings[:subset_size]
            input_ids = input_ids[:subset_size]
            target_ids = target_ids[:subset_size]

            # 3) Run free decoding
            free_acc = self.compare_free_decode_to_groundtruth(context_embeddings, target_ids)

            # 4) Log the free-decoding accuracy on the train set
            self.log("train_acc_free_decode", free_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        context_embeddings, input_ids, target_ids = batch
        attention_mask = (input_ids != self.pad_idx).long()

        # ---- Teacher-forced metrics ----
        logits = self(context_embeddings, input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        preds = torch.argmax(logits, dim=-1)
        mask = (target_ids != self.pad_idx)
        correct = ((preds == target_ids) & mask).sum().float()
        total = mask.sum().float().clamp_min(1.0)
        acc = correct / total

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc_teacher_forced", acc, prog_bar=True)

         # ---- Free-decoding metrics ----
        # We decode from embeddings alone & compare to target_ids
        free_acc = self.compare_free_decode_to_groundtruth(context_embeddings, target_ids)
        self.log("val_acc_free_decode", free_acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        context_embeddings, input_ids, target_ids = batch
        attention_mask = (input_ids != self.pad_idx).long()

        logits = self(context_embeddings, input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        preds = torch.argmax(logits, dim=-1)
        mask = (target_ids != self.pad_idx)
        correct = ((preds == target_ids) & mask).sum().float()
        total = mask.sum().float().clamp_min(1.0)
        acc = correct / total

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc_teacher_forced", acc, prog_bar=True)

        free_acc = self.compare_free_decode_to_groundtruth(context_embeddings, target_ids)
        self.log("test_acc_free_decode", free_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = steps_per_epoch * self.trainer.max_epochs

        # CosineAnnealingLR for the entire training
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=total_steps),
            "interval": "step",
            "frequency": 1,
            }
        return [optimizer], [scheduler]

        # arg_opt = self.config['optimizer']
        # optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        # arg_sche = AttrDict(self.config['schedular'])
        # scheduler, _ = create_scheduler(arg_sche, optimizer)
        # return [optimizer], [scheduler]

    #def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #    print('qqq', metric)

        ###########################
    # FREE DECODING FUNCTIONS #
    ###########################
    @torch.no_grad()
    def greedy_decode_batch(self, context_embeddings, max_length=150):
        """
        Greedily decode a batch of embeddings into SMILES tokens (no teacher forcing).
        
        Args:
            context_embeddings: [batch_size, embed_dim], the learned embeddings
            max_length: maximum decoding steps

        Returns:
            A list of token lists, each token list is the model-predicted sequence (including <sos>/<eos> if it generates them).
        """
        device = context_embeddings.device
        batch_size = context_embeddings.size(0)

        # Possibly adapt dimension
        if self.use_linear and self.encode_prefix and self.decode_prefix:
            context_embeddings = self.decode_prefix(self.encode_prefix(context_embeddings))

        # Start token ID
        start_idx = self.vocab.start_idx  # or whichever ID is <sos>
        end_idx   = self.vocab.end_idx    # or whichever ID is <eos>

        # Initialize partial sequences with <sos>
        partial_seqs = torch.full((batch_size, 1), fill_value=start_idx, dtype=torch.long, device=device)

        # Keep track of "finished" for samples that produce <eos> early
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):
            seq_len = partial_seqs.size(1)

            # Prepare cross-attention: expand context_embeddings to [batch, seq_len, hidden_dim]
            encoder_hidden_states = context_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

            # Forward
            outputs = self.model(
                input_ids=partial_seqs,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
                is_decoder=True,
            )
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # We only need the last step's logits
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            next_tokens = torch.argmax(last_logits, dim=-1)  # [batch_size]

            # Append to partial sequence
            partial_seqs = torch.cat([partial_seqs, next_tokens.unsqueeze(1)], dim=1)

            # Mark as finished if <eos> is generated
            newly_finished = (next_tokens == end_idx)
            finished = finished | newly_finished
            # if all finished, we can break early
            if torch.all(finished):
                break

        # Convert partial_seqs to a list of token IDs, removing <sos> if desired
        token_sequences = partial_seqs.tolist()
        # remove the first <sos> if you prefer
        # e.g., sequence[1:] until we hit <eos>
        final_sequences = []
        for seq in token_sequences:
            # find the first <eos> if it exists
            if end_idx in seq:
                eos_pos = seq.index(end_idx)
                seq = seq[:eos_pos+1]  # keep <eos> or cut it, up to you
            final_sequences.append(seq)
        return final_sequences

    def compare_free_decode_to_groundtruth(self, context_embeddings, target_ids):
        """
        Decodes each sample from 'context_embeddings' (free decoding),
        compares to 'target_ids' (the ground truth tokens).

        For accuracy, we do an exact match comparison at the token level
        (excluding <pad> and ignoring length differences for now).

        We also print out a few samples of predicted vs. ground truth
        SMILES strings for inspection.
        """
        device = context_embeddings.device
        decoded_seqs = self.greedy_decode_batch(context_embeddings)

        # Convert target_ids -> lists of IDs
        # Each row of target_ids might contain <pad> after <eos>.
        # We'll truncate at <eos> for a fair comparison.
        ground_truth_seqs = []
        end_idx = self.vocab.end_idx
        for row in target_ids.cpu().tolist():
            # Truncate at <eos> if it exists
            if end_idx in row:
                eos_pos = row.index(end_idx)
                row = row[:eos_pos + 1]
            ground_truth_seqs.append(row)

        # Now compute fraction of EXACT matches
        # i.e. the predicted sequence of tokens == ground-truth sequence of tokens
        num_correct = 0
        rounds = 0
        for pred_ids, gt_ids in zip(decoded_seqs, ground_truth_seqs):
            # Convert token IDs -> token strings -> a single SMILES string
            pred_tokens = self.vocab.decode(pred_ids)
            gt_tokens = self.vocab.decode(gt_ids)

            # Join them into a single SMILES if your vocab tokens are single characters
            # or otherwise adapt if tokens need spacing, etc.
            pred_smiles = "".join(pred_tokens)
            gt_smiles = "".join(gt_tokens)

            # Print a few samples for debugging
            if rounds < 7:  # e.g. first 7
                print(f"\n[Sample {rounds + 1}]")
                print(f"Predicted SMILES:  {pred_smiles}")
                print(f"Ground Truth SMILES:  {gt_smiles}")
                rounds += 1

            # Check if the raw ID lists match exactly
            if pred_ids == gt_ids:
                num_correct += 1

        acc = num_correct / len(decoded_seqs)
        return acc


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
