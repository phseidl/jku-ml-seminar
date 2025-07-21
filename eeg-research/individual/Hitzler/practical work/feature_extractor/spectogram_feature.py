# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class SPECTROGRAM_FEATURE_BINARY2(nn.Module):
    def __init__(self,
                 sample_rate: int = 200,
                 frame_length: int = 16,
                 frame_shift: int = 8,
                 feature_extract_by: str = 'kaldi'):
        super(SPECTROGRAM_FEATURE_BINARY2, self).__init__()

        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 1

        if self.feature_extract_by == 'kaldi':
            import torchaudio
            self.transforms = torchaudio.transforms.Spectrogram(
                n_fft=self.freq_resolution * self.sample_rate,
                win_length=frame_length,
                hop_length=frame_shift
            )
        else:
            self.n_fft = self.freq_resolution * self.sample_rate
            self.hop_length = frame_shift
            self.frame_length = frame_length
            self.window = torch.hamming_window(frame_length)

    def forward(self, batch):
        if self.feature_extract_by == 'kaldi':
            stft = self.transforms(batch)
            amp = torch.log(torch.abs(stft) + 1e-10)
        else:
            device = batch.device
            self.window = self.window.to(device)

            # Process batch directly
            stft = torch.stft(
                batch,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.frame_length,
                window=self.window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True
            )
            amp = torch.log(torch.abs(stft) + 1e-10)

        # Limit spectrogram dimensions to [:100, :100]
        return amp[..., :100, :100]
