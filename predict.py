# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH


import shutil
import random

from tempfile import TemporaryDirectory
from distutils.dir_util import copy_tree
from typing import Optional, Iterator, List
from cog import BasePredictor, Input, Path, BaseModel
import torch
import datetime

# Model specific imports
import torchaudio
import subprocess
import typing as tp

from audiocraft.models import MusicGen
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write
from audiocraft.models import MultiBandDiffusion
from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python
madmom.audio.filters.np.float = float

import soundfile as sf
import librosa
import numpy as np
import pyrubberband as pyrb



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.medium_model = self._load_model(
        #     model_path=MODEL_PATH,
        #     cls=MusicGen,
        #     model_id="facebook/musicgen-medium",
        # )
        # self.medium_model = MusicGen.get_pretrained('facebook/musicgen-medium')
        # self.lofi_model = MusicGen.get_pretrained('/src/models/lofi')
        self.melody_model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")
        
        # self.large_model = self._load_model(
        #     model_path=MODEL_PATH,
        #     cls=MusicGen,
        #     model_id="facebook/musicgen-large",
        # )
        self.large_model = MusicGen.get_pretrained('facebook/musicgen-stereo-large')

        self.mbd = MultiBandDiffusion.get_mbd_musicgen()

        self.beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda:0",
        )

        

    def predict(
        self,
        prompt: str = Input(
            description="A description of the music you want to generate.",
            default=None
        ),
        bpm: float = Input(
            description="Tempo in beats per minute",
            default=90.0,
            ge=40,
            le=300,
        ),
        max_duration: int = Input(
            description="Maximum duration of the generated loop in seconds.",
            default=8,
            le=30,
            ge=2,
        ),
        model_version: str = Input(
            description="Model to use for generation. .",
            default="large",
            choices=["melody", "large","lofi"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=-1,
        ),
        use_multiband_diffusion: bool = Input(
            description="Use MultiBandDiffusion for decoding. Should be higher quality but slower..",
            default=True,
        ),
        audio_input: Path = Input(
            description="Audio file to be continued by the model.",
            default=None,
        ),
    ) -> List[Path]:
        if prompt:
            prompt = f", {bpm}bpm. 320kbps 48khz. {prompt}"
        if not prompt:
            prompt = None
        # model = self.medium_model if model_version == "medium" else self.large_model

        if model_version == "melody":
            model = self.melody_model
        elif model_version == "large":
            model = self.large_model
        elif model_version == "lofi":
            model = self.lofi_model

        model.set_generation_params(
            duration=max_duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2**32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        print("Generating variation 1")

        if audio_input:
            audio_prompt, sample_rate = torchaudio.load(audio_input)
            # normalize
            audio_prompt = audio_prompt / torch.abs(audio_prompt).max()
            audio_prompt_duration = len(audio_prompt[0]) / sample_rate
            
            multiplier = 1 if model_version == "melody" else 2
            model.set_generation_params(
                duration=audio_prompt_duration * multiplier,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                cfg_coef=classifier_free_guidance,
            )

            if model_version == "melody":
                wav, tokens = model.generate_with_chroma(
                    melody_wavs=audio_prompt,
                    melody_sample_rate=sample_rate,
                    descriptions=[prompt],
                    return_tokens=True,
                    progress=True,
                )
            else:
                descriptions = {"descriptions": [prompt] } if prompt else {}
                wav, tokens = model.generate_continuation(
                    prompt=audio_prompt,
                    prompt_sample_rate=sample_rate,
                    return_tokens=True,
                    progress=True,
                    **descriptions
                )
            
            # if use_multiband_diffusion:
            #     wav = self.mbd.tokens_to_wav(tokens)

            # wav = wav.cpu().detach().numpy()[0, 0]
            # # normalize
            # wav = wav / np.abs(wav).max()

            # start_time = 0
            # end_time = audio_prompt_duration * 2

            # actual_bpm = bpm

            # print(f"{start_time=}, {end_time=}")

        else:
            wav, tokens = model.generate([prompt], return_tokens=True, progress=True)
            
        if use_multiband_diffusion:
            left, right = model.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])
            wav = self.mbd.tokens_to_wav(tokens)

        wav = wav.cpu().detach().numpy()[0, 0]
        # normalize
        wav = wav / np.abs(wav).max()

        audio_duration = len(wav) / model.sample_rate

        beats = self.estimate_beats(wav, model.sample_rate)
        start_time, end_time = self.get_loop_points(beats)
        
        # shift to start 0 
        end_time = end_time - start_time
        start_time = 0

        # loop_seconds = end_time - start_time

        print("Beats:\n", beats)
        print(f"{start_time=}, {end_time=}")

        num_beats = len(beats[(beats[:, 0] >= start_time) & (beats[:, 0] < end_time)])
        duration = end_time - start_time
        actual_bpm = num_beats / duration * 60
        if (
            abs(actual_bpm - bpm) > 15
            and abs(actual_bpm / 2 - bpm) > 15
            and abs(actual_bpm * 2 - bpm) > 15
        ):
            # raise ValueError(
            #     f"Failed to generate a loop in the requested {bpm} bpm. Please try again."
            # )
            print("could not generate loop in requested bpm, returning as is")
            start_time = 0
            end_time = audio_duration
        else:
            # Allow octave errors
            if abs(actual_bpm / 2 - bpm) <= 10:
                actual_bpm = actual_bpm / 2
            elif abs(actual_bpm * 2 - bpm) <= 10:
                actual_bpm = actual_bpm * 2

        start_sample = int(start_time * model.sample_rate)
        end_sample = int(end_time * model.sample_rate)
        loop = wav[start_sample:end_sample]

        # # do a quick blend with the lead-in do avoid clicks
        # num_lead = 100
        # lead_start = start_sample - num_lead
        # lead = wav[lead_start:start_sample]
        # num_lead = len(lead)
        # loop[-num_lead:] *= np.linspace(1, 0, num_lead)
        # loop[-num_lead:] += np.linspace(0, 1, num_lead) * lead

        
        stretched = pyrb.time_stretch(loop, model.sample_rate, bpm / actual_bpm)

        outputs = []
        self.write(stretched, model.sample_rate, output_format, "out-0")
        outputs.append(Path("out-0.wav"))

        # if variations > 1:
        #     # Use last 4 beats as audio prompt
        #     last_4beats = beats[beats[:, 0] <= end_time][-5:]
        #     audio_prompt_start_time = last_4beats[0][0]
        #     audio_prompt_end_time = last_4beats[-1][0]
        #     audio_prompt_start_sample = int(audio_prompt_start_time * model.sample_rate)
        #     audio_prompt_end_sample = int(audio_prompt_end_time * model.sample_rate)
        #     audio_prompt_seconds = audio_prompt_end_time - audio_prompt_start_time
        #     audio_prompt = torch.tensor(
        #         wav[audio_prompt_start_sample:audio_prompt_end_sample]
        #     )[None]
        #     audio_prompt_duration = audio_prompt_end_sample - audio_prompt_start_sample

        #     model.set_generation_params(
        #         duration=loop_seconds + audio_prompt_seconds + 0.1,
        #         top_k=top_k,
        #         top_p=top_p,
        #         temperature=temperature,
        #         cfg_coef=classifier_free_guidance,
        #     )

        #     for i in range(1, variations):
        #         print(f"\nGenerating variation {i + 1}")

        #         continuation, tokens = model.generate_continuation(
        #             prompt=audio_prompt,
        #             prompt_sample_rate=model.sample_rate,
        #             descriptions=[prompt],
        #             return_tokens=True,
        #             progress=True,
        #         )
                
        #         if use_multiband_diffusion:
        #             continuation = self.mbd.tokens_to_wav(tokens)

        #         variation_loop = continuation.cpu().detach().numpy()[
        #             0, 0, audio_prompt_duration : audio_prompt_duration + len(loop)
        #         ]
        #         variation_loop[-num_lead:] *= np.linspace(1, 0, num_lead)
        #         variation_loop[-num_lead:] += np.linspace(0, 1, num_lead) * lead

        #         variation_stretched = pyrb.time_stretch(
        #             variation_loop, model.sample_rate, bpm / actual_bpm
        #         )
        #         # add_output(
        #         #     outputs,
        #         #     self.write(
        #         #         variation_stretched,
        #         #         model.sample_rate,
        #         #         output_format,
        #         #         f"out-{i}",
        #         #     ),
        #         # )
        #         self.write(
        #             variation_stretched,
        #             model.sample_rate,
        #             output_format,
        #             f"out-{i}",
        #         ) 
        #         outputs.append(Path(f"out-{i}.wav"))
        return outputs

    def estimate_beats(self, wav, sample_rate):
        # resample to BeatNet's sample rate
        beatnet_input = librosa.resample(
            wav,
            orig_sr=sample_rate,
            target_sr=self.beatnet.sample_rate,
        )
        return self.beatnet.process(beatnet_input)

    def get_loop_points(self, beats):
        # extract an even number of bars
        downbeat_times = beats[:, 0][beats[:, 1] == 1]
        num_bars = len(downbeat_times) - 1

        if num_bars < 1:
            raise ValueError(
                "Less than one bar detected. Try increasing max_duration, or use a different seed."
            )

        even_num_bars = int((num_bars // 4) * 4)
        #even_num_bars = int(2 ** np.floor(np.log2(num_bars)))
        if even_num_bars < 4:
            even_num_bars = 4
        print("even_num_bars", even_num_bars)        
        start_time = downbeat_times[0]
        end_time = downbeat_times[even_num_bars]

        return start_time, end_time

    def write(self, audio, sample_rate, output_format, name):
        wav_path = name + ".wav"
        sf.write(wav_path, audio, sample_rate)

        if output_format == "mp3":
            mp3_path = name + ".mp3"
            subprocess.call(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", wav_path, mp3_path]
            )
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)


def add_output(outputs, path):
    for i in range(1, 21):
        field = f"variation_{i:02d}"
        if getattr(outputs, field) is None:
            setattr(outputs, field, path)
            return
    raise ValueError("Failed to add output")


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
