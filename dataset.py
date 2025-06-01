import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from yml.option import parse
import librosa
import soundfile as sf
from utils import *
prompt_template import *

class Datasets(Dataset):
    def __init__(self, json_file1, json_file2, rir_directory, mix_count, sample_rate=16000):
        """
        Initialize dataset with JSON files, RIR directory, and mix_count.
        Args:
            json_file1 (str): Path to the first JSON file.
            json_file2 (str): Path to the second JSON file.
            rir_directory (str): Directory containing .npz RIR files.
            mix_count (int): Number of mixtures to generate.
            sample_rate (int): Sample rate for resampling.
        """
        
        self.json_file1 = json_file1
        self.json_file2 = json_file2
        self.mix_count = mix_count
        self.sample_rate = sample_rate
    
        # Load JSON data
        with open(json_file1, 'r', encoding='utf-8') as f1:
            self.data1 = json.load(f1)
        with open(json_file2, 'r', encoding='utf-8') as f2:
            self.data2 = json.load(f2)
        
        # Load RIR files
        self.rir_files = [os.path.join(rir_directory, f) for f in os.listdir(rir_directory) if f.endswith('.npz')]
        self.template = ShortTemplate()

        self.index = 0
        self.snr = 0
        self.overlap_duration = 0
        self.language = 0
        self.aligned_transcription = 0
        self.age_category = 0
        self.temporal_order = 0
        self.emotion = 0
        self.gender = 0
        self.speaking_rate_category = 0
        self.actual_duration_category = 0
        self.speaking_duration_category = 0
        self.mean_f0_dp_category = 0
        self.log_f0_range_dp_category = 0
        self.mean_f0_rvbt_category = 0
        self.log_f0_range_rvbt_category = 0
        self.loundness_category = 0
        self.distance_category = 0

        self.specific_language = []
        self.emotion_language = []
        self.emotion_gender = []
        self.specific_age_category = []
        self.age_language = []
        self.age_gender = []
        self.specific_temporal_order = []
        self.specific_emotion = []
        self.specific_gender = []
        self.specific_speaking_rate_category = []
        self.specific_actual_duration_category = []
        self.specific_speaking_duration_category = []
        self.specific_mean_f0_dp_category = []
        self.specific_log_f0_range_dp_category = []
        self.specific_mean_f0_rvbt_category = []
        self.specific_log_f0_range_rvbt_category = []
        self.specific_loundness_category = []
        self.specific_distance_category = []

    def __len__(self):
        """Return the total number of mixtures."""
        
        return self.mix_count

    def __getitem__(self, idx):
        """
        Generate a single data sample for mixing.
        Args:
            idx (int): Index of the data sample.
        Returns:
            dict: Contains mixed audio, target audio, interference audio, and metadata.
        """
        
        self.index += 1

        if 'train' in self.json_file1:
            sub_dir = 'train'
            rng = np.random.default_rng(np.random.PCG64(seed=idx))
            rng2 = np.random.default_rng(np.random.PCG64(seed=idx)) 
        elif 'dev' in self.json_file1:
            sub_dir = 'dev'
            rng = np.random.default_rng(np.random.PCG64(seed=idx))
            rng2 = np.random.default_rng(np.random.PCG64(seed=idx)) 
        elif 'test' in self.json_file1:
            sub_dir = 'test'
            rng = np.random.default_rng(np.random.PCG64(seed=idx))
            rng2 = np.random.default_rng(np.random.PCG64(seed=idx)) 
        
        output_dir = os.path.join("tseData", sub_dir)
        os.makedirs(output_dir, exist_ok=True)
        

        # Randomly select wav entries from the JSON files
        entry1 = rng.choice(self.data1)
        entry2 = rng.choice(self.data2)
        sr = 16000  # sampling rate of audio
        target_duration = 6 # target duration 6s

        # Load wav files and resample wavs if necessary
        wav1_file = entry1['file_path']
        wav1, sr = resample_wav(wav1_file, self.sample_rate)
        wav1, wav1_start_time_original, wav1_end_time_original, wav1_words, wav1_words_end_times = pad_or_trim_remove_silence(wav1, target_duration, wav1_file, sr, rng)

        wav2_file = entry2['file_path']
        wav2, sr = resample_wav(wav2_file, self.sample_rate)
        wav2, wav2_start_time_original, wav2_end_time_original, wav2_words, wav2_words_end_times = pad_or_trim_remove_silence(wav2, target_duration, wav2_file, sr, rng)

        wav1_filtered_duration, wav1_voice_duration, wav2_filtered_duration, wav2_voice_duration, overlap_duration, wav1, wav2, wav1_start_time_mixture, wav1_end_time_mixture, wav2_start_time_mixture, wav2_end_time_mixture, wav1_start_time_original, wav1_end_time_original, wav2_start_time_original, wav2_end_time_original, wav1_words, wav1_end_times, wav2_words, wav2_end_times = overlap_wavs(rng, entry1, entry2, wav1, wav1_start_time_original, wav2, wav2_start_time_original, target_duration, sr, wav1_words, wav1_words_end_times, wav2_words, wav2_words_end_times)
        
        # Load RIRs and apply
        rir_file = rng.choice(self.rir_files)
        rirs = np.load(rir_file)
        rir1 = rirs['rir1']
        rir1_dp = rirs['rir1_dp']
        rir2 = rirs['rir2']
        rir2_dp = rirs['rir2_dp']
        rt60 = float(os.path.basename(rir_file).rstrip('.npz').split('_')[1])
        distance1 = float(os.path.basename(rir_file).rstrip('.npz').split('_')[2])
        distance2 = float(os.path.basename(rir_file).rstrip('.npz').split('_')[3])

        sir = rng.uniform(0, 6)
        # Initialize random number generators
        wav1_rvbt, wav1_dp = apply_rir_to_wav(wav1, rir1, rir1_dp)
        
        wav1_rvbt_distance = distance1
        
        wav1_dp_distance = distance1

        wav2_rvbt, wav2_dp = apply_rir_to_wav(wav2, rir2, rir2_dp)
        
        wav2_rvbt_distance = distance2
        
        wav2_dp_distance = distance2

        wav1_time_diff = abs((wav1_end_time_original - wav1_start_time_original) - (wav1_end_time_mixture - wav1_start_time_mixture))
        wav2_time_diff = abs((wav2_end_time_original - wav2_start_time_original) - (wav2_end_time_mixture - wav2_start_time_mixture))
        
        assert wav1_time_diff < 0.01
        assert wav2_time_diff < 0.01

        wav1_actual_length = int((wav1_end_time_mixture - wav1_start_time_mixture)*sr)
        wav2_actual_length = int((wav2_end_time_mixture - wav2_start_time_mixture)*sr)
        
        # Mix the wavs
        mixed_wav, wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir = mix_wavs(wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir, wav1_actual_length, wav2_actual_length, wav1_filtered_duration, wav2_filtered_duration)

        wav1_rvbt_sir = round(sir, 2) if sir is not None else None
        wav1_dp_sir = wav1_rvbt_sir
        wav2_rvbt_sir = -wav1_rvbt_sir if wav1_rvbt_sir is not None else None
        wav2_dp_sir = wav2_rvbt_sir

        # compute mean f0 and f0 range
        mean_f0_wav1_dp, log_f0_range_wav1_dp = calculate_f0_features(wav1_dp, sr, wav1_voice_duration)
        mean_f0_wav2_dp, log_f0_range_wav2_dp = calculate_f0_features(wav2_dp, sr, wav2_voice_duration)

        time_diff = wav1_start_time_mixture - wav2_start_time_mixture
        if time_diff <= -0.1:
            wav1_temporal_order, wav2_temporal_order = 'first', 'second'
        elif abs(time_diff) < 0.1:
            wav1_temporal_order, wav2_temporal_order = 'similar', 'similar'
        else:
            wav1_temporal_order, wav2_temporal_order = 'second', 'first'
        
        # Randomly select target and interference
        if random.choice([True, False]): # if wav1 is target, i.e., speaker 1 appears first in the mixture
            target_wav_rvbt = wav1_rvbt
            interference_wav_rvbt = wav2_rvbt
            target_wav_dp = wav1_dp
            interference_wav_dp = wav2_dp
            target_entry = entry1
            interference_entry = entry2
            target_entry['actual_duration'] = round(wav1_actual_length / sr, 2)
            interference_entry['actual_duration'] = round(wav2_actual_length / sr, 2)
            target_entry['filtered_duration'] = round(wav1_filtered_duration, 2)
            interference_entry['filtered_duration'] = round(wav2_filtered_duration, 2)
            target_entry['voice_duration'] = round(wav1_voice_duration, 2)
            interference_entry['voice_duration'] = round(wav2_voice_duration, 2)
            target_entry['start_time_mixture'] = round(wav1_start_time_mixture, 2)
            target_entry['end_time_mixture'] = round(wav1_end_time_mixture, 2)
            interference_entry['start_time_mixture'] = round(wav2_start_time_mixture, 2)
            interference_entry['end_time_mixture'] = round(wav2_end_time_mixture, 2)
            target_entry['start_time_original'] = round(wav1_start_time_original, 2)
            interference_entry['start_time_original'] = round(wav2_start_time_original, 2)
            target_entry['aligned_transcription'] = wav1_words  
            interference_entry['aligned_transcription'] = wav2_words  
            target_entry['aligned_transcription_endtimes'] = wav1_end_times
            interference_entry['aligned_transcription_endtimes'] = wav2_end_times
            target_entry['mean_f0_dp'] = mean_f0_wav1_dp
            target_entry['log_f0_range_dp'] = log_f0_range_wav1_dp
            target_entry['mean_f0_rvbt'] = mean_f0_wav1_dp
            target_entry['log_f0_range_rvbt'] = log_f0_range_wav1_dp
            interference_entry['mean_f0_dp'] = mean_f0_wav2_dp
            interference_entry['log_f0_range_dp'] = log_f0_range_wav2_dp
            interference_entry['mean_f0_rvbt'] = mean_f0_wav2_dp
            interference_entry['log_f0_range_rvbt'] = log_f0_range_wav2_dp
            target_entry['sir'] = wav1_rvbt_sir
            target_entry['rt60'] = round(rt60, 3)
            target_entry['distance'] = round(wav1_rvbt_distance, 2)
            interference_entry['sir'] = wav2_rvbt_sir
            interference_entry['rt60'] = round(rt60, 3)
            interference_entry['distance'] = round(wav2_rvbt_distance, 2)
            target_entry['temporal_order'] = wav1_temporal_order
            interference_entry['temporal_order'] = wav2_temporal_order
            
            
        else: # if wav2 is target, i.e., speaker 2 appears second in the mixture
            target_wav_rvbt = wav2_rvbt
            interference_wav_rvbt = wav1_rvbt
            target_wav_dp = wav2_dp
            interference_wav_dp = wav1_dp
            target_entry = entry2
            interference_entry = entry1
            target_entry['actual_duration'] = round(wav2_actual_length / sr, 2)
            interference_entry['actual_duration'] = round(wav1_actual_length / sr, 2)
            target_entry['filtered_duration'] = round(wav2_filtered_duration, 2)
            interference_entry['filtered_duration'] = round(wav1_filtered_duration, 2)
            target_entry['voice_duration'] = round(wav2_voice_duration, 2)
            interference_entry['voice_duration'] = round(wav1_voice_duration, 2)
            target_entry['start_time_mixture'] = round(wav2_start_time_mixture, 2)
            target_entry['end_time_mixture'] = round(wav2_end_time_mixture, 2)
            interference_entry['start_time_mixture'] = round(wav1_start_time_mixture, 2)
            interference_entry['end_time_mixture'] = round(wav1_end_time_mixture, 2)
            target_entry['start_time_original'] = round(wav2_start_time_original, 2)
            interference_entry['start_time_original'] = round(wav1_start_time_original, 2)
            target_entry['aligned_transcription'] = wav2_words
            interference_entry['aligned_transcription'] = wav1_words 
            target_entry['aligned_transcription_endtimes'] = wav2_end_times
            interference_entry['aligned_transcription_endtimes'] = wav1_end_times
            target_entry['mean_f0_dp'] = mean_f0_wav2_dp
            target_entry['log_f0_range_dp'] = log_f0_range_wav2_dp
            target_entry['mean_f0_rvbt'] = mean_f0_wav2_dp
            target_entry['log_f0_range_rvbt'] = log_f0_range_wav2_dp
            interference_entry['mean_f0_dp'] = mean_f0_wav1_dp
            interference_entry['log_f0_range_dp'] = log_f0_range_wav1_dp
            interference_entry['mean_f0_rvbt'] = mean_f0_wav1_dp
            interference_entry['log_f0_range_rvbt'] = log_f0_range_wav1_dp
            target_entry['sir'] = wav2_rvbt_sir
            target_entry['rt60'] = round(rt60, 3)
            target_entry['distance'] = round(wav2_rvbt_distance, 2)
            interference_entry['sir'] = wav1_rvbt_sir
            interference_entry['rt60'] = round(rt60, 3)
            interference_entry['distance'] = round(wav1_rvbt_distance, 2)
            target_entry['temporal_order'] = wav2_temporal_order
            interference_entry['temporal_order'] = wav1_temporal_order
        
        if target_entry.get('transcription_id') is None:
            target_entry['transcription_id'] = None

        if interference_entry.get('transcription_id') is None:
            interference_entry['transcription_id'] = None
        
        if target_entry.get('transcription') is None:
            target_entry['transcription'] = None

        if interference_entry.get('transcription') is None:
            interference_entry['transcription'] = None
        
        if target_entry.get('accent') is None:
            target_entry['accent'] = None

        if interference_entry.get('accent') is None:
            interference_entry['accent'] = None
        
        if target_entry.get('emotion') is None:
            target_entry['emotion'] = None

        if interference_entry.get('emotion') is None:
            interference_entry['emotion'] = None

        if target_entry.get('age') is None:
            target_entry['age'] = None

        if interference_entry.get('age') is None:
            interference_entry['age'] = None

        # Analyze target_dentry's word boundary using the TextGrid file in the same directory as the .wav file
        textgrid_file = os.path.splitext(target_entry['file_path'])[0] + ".TextGrid"
  
        if os.path.isfile(textgrid_file):
            # If the TextGrid file exists, parse and update word_alignment
            words, end_times = parse_textgrid(textgrid_file)
            word_alignment = format_word_alignment(words, end_times)
            target_entry["original_word_alignment"] = word_alignment
            target_entry["original_words_duration"] = round(float(end_times[-1]) - float(end_times[0]), 2)

        else:
            # If the TextGrid file does not exist, set word_alignment to None
            target_entry["original_word_alignment"] = None
            target_entry["original_words_duration"] = None

        # Analyze interference_entry's word boundary using the TextGrid file in the same directory as the .wav file
        textgrid_file = os.path.splitext(interference_entry['file_path'])[0] + ".TextGrid"
        
        if os.path.isfile(textgrid_file):
            # If the TextGrid file exists, parse and update word_alignment
            words, end_times = parse_textgrid(textgrid_file)
            word_alignment = format_word_alignment(words, end_times)
            interference_entry["original_word_alignment"] = word_alignment
            interference_entry["original_words_duration"] = round(float(end_times[-1]) - float(end_times[0]), 2)
        else:
            # If the TextGrid file does not exist, set word_alignment to None
            interference_entry["original_word_alignment"] = None
            interference_entry["original_words_duration"] = None
            
        # randomly select enrollment audio
        enroll_file = random_wav(target_entry, rng)
        speaking_rate_target = calculate_speaking_rate(target_entry['aligned_transcription'], target_entry['filtered_duration'] , target_entry['language'])

        # Add speaking_rate cues to target_entry
        if speaking_rate_target is not None:
            target_entry['speaking_rate'] = round(speaking_rate_target, 2)
        else:
            target_entry['speaking_rate'] = None

        speaking_rate_interference = calculate_speaking_rate(interference_entry['aligned_transcription'], interference_entry['filtered_duration'] , interference_entry['language'])

        # Add speaking_rate cues to interference_entry
        if speaking_rate_interference is not None:
            interference_entry['speaking_rate'] = round(speaking_rate_interference, 2)
        else:
            interference_entry['speaking_rate'] = None

        # Json naming
        mixture_json_filename = f"{idx}_mixture.json"
        
        def save_audio_and_metadata(idx, output_dir, mixed_wav, target_wav_rvbt, target_wav_dp, enroll_file, rir_files, rng):
            """Save mixed and processed audio into a single file to save memory."""

            # Combine all audio into a dictionary for saving in one file
            audio_data = {
                'mixed': mixed_wav,
                'target_rvbt': target_wav_rvbt,
                'target_dp': target_wav_dp
            }

            if enroll_file is not None:
                enroll_wav, sr = resample_wav(enroll_file, self.sample_rate)
                enroll_wav = pad_audio(enroll_wav, rng, sr, mixed_wav.shape[0] / sr)

                # Load RIRs and apply
                rir_file = rng2.choice(rir_files)
                rirs = np.load(rir_file)
                rir = rirs['rir1']
                rir_dp = rirs['rir1_dp']

                # Randomly select RIRs
                rir_list = [(rir, rir_dp), (rirs['rir1'], rirs['rir1_dp']), (rirs['rir2'], rirs['rir2_dp'])]
                index = rng.choice(len(rir_list))
                enroll_wav_rir, enroll_wav_rir_dp = rir_list[index]

                # Apply selected RIR and RIR_DP to enroll_wav
                enroll_wav_rvbt, enroll_wav_dp = apply_rir_to_wav(enroll_wav, enroll_wav_rir, enroll_wav_rir_dp)

                audio_data['enroll_rvbt'] = enroll_wav_rvbt
                audio_data['enroll_dp'] = enroll_wav_dp
            else:
                print('no enrollment speech')

            # Save all audio data into a single file
            combined_audio_file = os.path.join(output_dir, f"{idx}_audio.npz")
            np.savez(combined_audio_file, **audio_data)

        save_audio_and_metadata(idx, output_dir, mixed_wav, target_wav_rvbt, target_wav_dp, enroll_file, self.rir_files, rng)

        target_entry, interference_entry = combine_target_and_interference(target_entry, interference_entry, output_dir, mixture_json_filename)

        cues = generate_cues(target_entry, interference_entry)

        if not cues:
            print('no cues')
            prompt_manual = 'no cues'

        else:
            target_style = build_short_style_descriptions_for_one_speaker(target_entry, cues, rng)
            acts = ['1']
            spks = [target_style]
            prompt_manual = self.template(acts, spks, rng) 

        mixture_json_data = {
            'target_entry': target_entry,
            'interference_entry': interference_entry,
            'prompt_manual': prompt_manual,
        }

        # save json file
        save_json(mixture_json_data, os.path.join(output_dir, mixture_json_filename))

        # Package metadata
        metadata = {
            "target_entry": target_entry,
            "interference_entry": interference_entry
        }

        return {
            "metadata": metadata
        }

def collate_fn(batch):
    """
    Custom collate function to combine data samples into a batch.
    Args:
        batch (list): List of data samples.
    Returns:
        dict: Batched data.
    """
    metadata = [item["metadata"] for item in batch]
    return {
        "metadata": metadata
    }

def make_dataloader(is_train=False,
                    data_kwargs=None,
                    num_workers=4,
                    batch_size=16):
    dataset = Datasets(**data_kwargs)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=DistributedSampler(dataset, shuffle=is_train),  # use DistributedSampler
                      num_workers=num_workers,
                      #shuffle=is_train,
                      collate_fn=collate_fn,
                      drop_last=False, pin_memory=True)


def setup_distributed(backend='gloo', init_method='env://'):
    dist.init_process_group(
        backend=backend,  # backend
        init_method=init_method  
    )



if __name__ == "__main__":
    # Set distribution enviroment to speed up simulate
    os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # local ip
    os.environ["MASTER_PORT"] = "29500"      # port
    setup_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False)

    # Dataset and DataLoader

    train_dataloader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['train'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['batch_size'])
    val_dataloader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['val'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['batch_size'])
    test_dataloader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['test'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['batch_size'])

    # Iterate through DataLoader
    for batch_idx, batch in enumerate(train_dataloader):
        print('batch_idx', batch_idx)
    for batch_idx, batch in enumerate(val_dataloader):
        print('batch_idx', batch_idx)
    for batch_idx, batch in enumerate(test_dataloader):
        print('batch_idx', batch_idx)

