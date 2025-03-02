import torch
from tqdm import tqdm
import copy
import wandb


class Trainer:
    def __init__(self, model, optimizer, loss_fn, accelerator, device="cpu", use_wandb=False, wandb_project_name="UPT4EEG"):
        """
        Initialize the Trainer.

        Args:
            model: The PyTorch model to train.
            optimizer: Optimizer for updating model weights.
            loss_fn: Loss function for training.
            device: Device to run the training on (e.g., "cpu" or "cuda").
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(self.device)
        self.use_wandb = use_wandb
        self.wandb_project_name = wandb_project_name
        self.accelerator = accelerator

    def train(self, train_dataloader, val_dataloader, num_epochs, train_config):
        """
        Train the model.

        Args:
            dataloader: DataLoader for the training data.
            num_epochs: Number of epochs to train.
            log_interval: How often (in batches) to log training progress.
        """
        total_updates = train_config['total_updates']
        ckpts = train_config['ckpts']
        lrs = train_config['lrs']
        wandb_config = train_config['wandb_config']
        accumulation_steps = train_config['accumulation_steps']

        if self.use_wandb:
            run = wandb.init(
                project=self.wandb_project_name,
                config=wandb_config,
            )

        # train model
        update = 0
        pbar = tqdm(total=total_updates)
        pbar.update(0)
        pbar.set_description("train_loss: ????? train_accuracy: ????% test_accuracy: ????%")
        test_accuracy = 0.0
        train_losses = []
        test_losses = []
        loss = None
        test_loss = None
        for epoch in range(num_epochs):
            # train for an epoch
            for i, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.model.train()  # Set model to training mode
                    # schedule learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lrs[update]
                    # forward pass
                    y_hat = self.model(
                        input_feat=batch["input_feat"].to(self.device),
                        input_pos=batch["input_pos"].to(self.device),
                        batch_idx=batch["batch_idx"].to(self.device),
                        output_pos=batch["output_pos"].to(self.device),
                    )
                    y = batch["target_feat"].to(self.device)
                    assert y_hat.shape == y.shape
                    loss = self.loss_fn(y_hat, y)

                    # backward pass
                    self.accelerator.backward(loss)

                    # update step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # status update
                    update += 1
                    pbar.update()
                    if test_loss is None:
                        pbar.set_description(f"train_loss: {loss.item():.4f}")
                    else:
                        pbar.set_description(
                            f"train_loss: {loss.item():.4f} "
                            f"test_loss: {test_loss:.4f} "
                        )
                    train_losses.append(loss.item())

                    if self.use_wandb:
                        run.log({
                            "update": update,
                            "training_loss": loss.item(),
                            "lr": param_group["lr"],
                            })

            # evaluate
            test_loss = 0.
            for batch in val_dataloader:
                self.model.eval()  
                with torch.no_grad():
                    y_hat = self.model(
                        input_feat=batch["input_feat"].to(self.device),
                        input_pos=batch["input_pos"].to(self.device),
                        batch_idx=batch["batch_idx"].to(self.device),
                        output_pos=batch["output_pos"].to(self.device),
                    )
                y = batch["target_feat"].to(self.device)
                test_loss += (self.loss_fn(y_hat, y)).item()
            test_loss /= len(val_dataloader)
            test_losses.append(test_loss)
            pbar.set_description(
                f"train_loss: {loss.item():.4f} "
                f"test_loss: {test_loss:.4f} "
            )

            if self.use_wandb:
                run.log({
                        "validation_loss": test_loss,
                        })

            state = dict(
                epoch=epoch + 1, min_loss=ckpts[0].bound, min_vloss=ckpts[1].bound,
                state_dict=self.model.state_dict(), loss=loss, val_loss=test_loss, learning_rate=param_group["lr"]
            )

            for ckpt in ckpts:
                improved_flag = ckpt.on_epoch_end(epoch + 1, state)
                if improved_flag:
                    best_model_state = copy.deepcopy(self.model.state_dict())

        pbar.close()

    def evaluate(self, dataloader):
        """
        Evaluate the model.

        Args:
            dataloader: DataLoader for the evaluation data.

        Returns:
            The average loss over the evaluation dataset.
        """
        self.model.eval()  # Set model to evaluation mode

        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss
