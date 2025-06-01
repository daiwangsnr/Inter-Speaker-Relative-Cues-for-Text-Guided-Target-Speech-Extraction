from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torchaudio
from pathlib import Path
import logging
from prompt_template import *


def generate_cues(target_entry, interference_entry):
    cues = []
    
    if target_entry['aligned_transcription'] is not None:
        # Extract cues from the target_entry, ignoring cues with attribute value 'similar'
        aligned_transcription = target_entry["aligned_transcription"].strip()
        cues.append('aligned_transcription')
        
    if target_entry['language'] is not None and target_entry['language'] != interference_entry['language'] :
        cues.append("language")

    if target_entry.get('age_category') is not None and interference_entry.get('age_category') is not None and target_entry.get('age_category') != interference_entry.get('age_category'):
        cues.append("age_category")

    if (target_entry.get('emotion') is not None and interference_entry.get('emotion') is not None and (target_entry.get('emotion')!='boredom') and target_entry.get('emotion') != interference_entry.get('emotion')):
        cues.append("emotion")

    if target_entry.get('gender') is not None and interference_entry.get('gender') is not None and target_entry.get('gender') != interference_entry.get('gender'):
        cues.append("gender")

    if target_entry['speaking_rate_category'] is not None and target_entry['speaking_rate_category'] != 'similar':
        cues.append("speaking_rate_category")

    if target_entry['speaking_duration_category'] is not None and target_entry['speaking_duration_category'] != 'similar':
        cues.append("speaking_duration_category")

    if target_entry['mean_f0_rvbt_category'] is not None and target_entry['mean_f0_rvbt_category'] != 'similar':
        cues.append("mean_f0_rvbt_category")

    if target_entry['log_f0_range_rvbt_category'] is not None and target_entry['log_f0_range_rvbt_category'] != 'similar':
        cues.append("log_f0_range_rvbt_category")

    if target_entry['loundness_category'] is not None and target_entry['loundness_category'] != 'similar':
        cues.append("loundness_category")

    if target_entry['distance_category'] is not None and target_entry['distance_category'] != 'similar':
        cues.append("distance_category")

    if target_entry['temporal_order'] is not None and target_entry['temporal_order'] != 'similar':
        cues.append("temporal_order")

    return cues

def resample_wav(wav_path, target_sr=16000):
    """Load and resample a audio file to the target sample rate."""
    try:
        wav, sr = librosa.load(wav_path, sr=None, mono=False) 
        # If original wav is a multi-channel audio, take the first channel
        if wav.ndim > 1:
            wav = wav[0, :]
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav, target_sr
    except Exception as e:
        logging.error(f"Error in resampling {wav_path}: {e}")
        raise

def calculate_total_duration(filtered_words, filtered_end_times, start_time_original):
    
    total_duration1 = 0 # Initialize a variable to compute filtered_duration
    total_duration2 = 0 # Initialize a variable to compute voice_duration
    last_valid_end_time = 0
    
    # Traverse each word and its corresponding end time
    for i, word in enumerate(filtered_words):

        current_end_time = filtered_end_times[i]

        if word == "":  # Detecting pauses
            # Calculate pause time
            pause_duration = current_end_time - last_valid_end_time
            if pause_duration <= 0.6:  # Retain pauses <= 0.6 seconds
                total_duration1 += pause_duration
            
            last_valid_end_time = current_end_time
        else:
            # Update total durations
            total_duration1 += current_end_time - last_valid_end_time
            total_duration2 += current_end_time - last_valid_end_time
            last_valid_end_time = current_end_time
    # filtered_duration means only remove silence > 0.6s in audio, voice_duration means remove all silence
    if filtered_end_times != []:
        filtered_duration, voice_duration = total_duration1-filtered_end_times[0], total_duration2-filtered_end_times[0]
    else:
        filtered_duration, voice_duration = 0, 0

    return filtered_duration, voice_duration

def compute_active_speech_duration(wav_timestamps):
    """
    Remove long silent segments based on the speech timestamp output by VAD, and calculate the total duration of the audio after removal.
    
    Args:
    - wav_timestamps (list of dict): each dict includes the 'start' and 'end' timestamps of VAD output.
    - total_duration (float): total duration of audio.

    Returns:
    - processed_duration (float): Total duration after removing long silent segments.
    """
 
    active_speech_duration1 = 0
    active_speech_duration2 = 0
    # Traverse the timestamp of each speech segment and calculate the total duration of the speech segment
    for i, segment in enumerate(wav_timestamps):
    
        start = segment['start']
        end = segment['end']

        # Accumulate the duration of speech segments
        active_speech_duration1 += (end - start)
        active_speech_duration2 += (end - start)

        # 如果存在下一个片段，计算当前片段结束与下一个片段开始之间的静音时长
        if i < len(wav_timestamps) - 1:
            next_start = wav_timestamps[i + 1]['start']
            silence_duration = next_start - end

            # 如果静音时长小于0.6秒，将其累加到总时长（将片段连接）
            if silence_duration <= 0.6:
                active_speech_duration1 += silence_duration

    return active_speech_duration1, active_speech_duration2

def parse_textgrid(textgrid_file):
    try:
        # 使用 UTF-8 读取文件
        with open(textgrid_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 只解析 item [1] 部分
        item_1_pattern = r'item \[1\]:\s*class = "IntervalTier".*?intervals: size = \d+\s*(.*?)item \[2\]'
        item_1_match = re.search(item_1_pattern, content, re.DOTALL)

        if not item_1_match:
            raise ValueError(f"Item [1] not found in the file: {textgrid_file}")

        intervals_section = item_1_match.group(1)

        # 使用正则表达式提取 intervals 和时间边界
        intervals_pattern = r'intervals \[.*?\]:\s*xmin = (.*?)\s*xmax = (.*?)\s*text = "(.*?)"'
        intervals = re.findall(intervals_pattern, intervals_section, re.DOTALL)

        words = []
        end_times = []

        for xmin, xmax, word in intervals:
            words.append(word)
            end_times.append(xmax)  # xmax 转为浮点数，便于后续计算

        return words, end_times

    except UnicodeDecodeError:
        raise RuntimeError(f"Failed to decode the file: {textgrid_file}. Please check the file encoding.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while parsing the file: {textgrid_file}. Error: {e}")

def remove_silence(wav, start_time_original, wav_file, sr):

    # Get the corresponding TextGrid file
    textgrid_file = os.path.splitext(wav_file)[0] + ".TextGrid"
    
    if os.path.isfile(textgrid_file):
        # Parse TextGrid for words and their end times
        words, end_times = parse_textgrid(textgrid_file)

        # Adjust start_time_original and calculate end_time_original
        end_time_original = start_time_original + len(wav) / sr

        # initial value
        start_time_original_processed = start_time_original
        end_time_original_processed = end_time_original

        # Filter words and end times to remove leading and trailing silence
        filtered_words = []
        filtered_end_times = []
        
        if words == [] or end_times == []:
            logging.error(f"empty!")
            
        for word, end_time in zip(words, end_times):
            if start_time_original <= float(end_time) <= end_time_original:
                filtered_words.append(word)
                filtered_end_times.append(float(end_time))

        # Remove leading silence (assume silence marked as ',')
        while filtered_words and filtered_words[0] == '':
            filtered_words.pop(0)
            filtered_end_times.pop(0)
            if filtered_end_times:
                start_time_original_processed = filtered_end_times[0]

        # Remove trailing silence (assume silence marked as ',')
        while filtered_words and filtered_words[-1] == '':
            filtered_words.pop(-1)
            filtered_end_times.pop(-1)
            if filtered_end_times:
                end_time_original_processed = filtered_end_times[-1]
        
        if filtered_words == [] or filtered_end_times == []:
            logging.error(f"empty!")
 
        # Actual start and end time of wav
        start_time = start_time_original_processed - start_time_original
        end_time = start_time + (end_time_original_processed - start_time_original_processed)
        
        # Slice wav based on updated time bounds
        wav = wav[int(start_time * sr):int(end_time * sr)]

        return wav, start_time_original_processed, end_time_original_processed, filtered_words, filtered_end_times


    else:
        # Apply Silero VAD to wav1 and wav2, removing the front and end silence in each of them.
        model = load_silero_vad()
        #wav = read_audio('path_to_audio_file')
        wav_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate = sr,
        return_seconds=True  # Return speech timestamps in seconds (default is samples)
        )
        # actual start and end time of wav
        start_time = wav_timestamps[0]['start']
        #remove trailing silence 
        end_time = wav_timestamps[-1]['end']
        # # do not remove trailing silence
        # end_time = len(wav)/sr
        wav = wav[int(start_time*sr):int(end_time*sr)]
        # compute original start and end time
        start_time_original_processed = start_time_original + start_time
        end_time_original_processed = start_time_original_processed + (end_time - start_time)

        filtered_words = []
        filtered_end_times = []

        return wav, start_time_original_processed, end_time_original_processed, filtered_words, filtered_end_times
  
def pad_or_trim_remove_silence(wav, target_duration, wav_file, sr, rng):
    """Pad or trim the audio file to the target duration and then remove silence."""
    target_len = int(target_duration * sr)
    
    if len(wav) >= target_len:
        filtered_words = []
        filtered_end_times = []
        max_retries = 50
        retries = 0
        while (filtered_words == [] or filtered_end_times == []) and retries < max_retries:
            start_idx = rng.integers(0, len(wav) - target_len + 1)
            wav_ = wav[start_idx:start_idx + target_len]
            start_time_original = start_idx / sr
            try:
                wav_, start_time_original, end_time_original, filtered_words, filtered_end_times = remove_silence(
                    wav_, start_time_original, wav_file, sr
                )
            except Exception as e:
                logging.error(f"Error during silence removal: {e}")
                filtered_words, filtered_end_times = [], []

            retries += 1

        if retries == max_retries:
            #raise ValueError("Failed to process audio segment after multiple retries.")
            filtered_words = []
            filtered_end_times == []

    else:
        start_time_original = 0
        # remove silence in short segement
        wav_, start_time_original, end_time_original, filtered_words, filtered_end_times = remove_silence(wav, start_time_original, wav_file, sr)
        
    return wav_, start_time_original if start_time_original is not None else None, end_time_original if end_time_original is not None else None, filtered_words, filtered_end_times

def apply_rir_to_wav(wav, rir,rir_dp):
    """Apply the room impulse response to a waveform."""

    wav_rvbt = np.convolve(wav, rir, mode='full')  # Applying the RIR
    wav_dp = np.convolve(wav, rir_dp, mode='full')  # Applying the RIR
    delay = np.argmax(rir)
    wav_rvbt = wav_rvbt[delay:delay + wav.shape[0]]
    wav_dp = wav_dp[delay:delay + wav.shape[0]]
    return wav_rvbt, wav_dp

def adjust_wav(wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir_db, wav1_length, wav2_length, wav1_filtered_duration, wav2_filtered_duration):
    """Adjust the loudness of wav1 and wav2 to achieve the desired SIR in dB."""
    # Calculate the RMS (power) of both waveforms use speech activity length
    sr = 16000
    if int(wav1_filtered_duration*sr) == 0:
        #print('wav1_length', wav1_length)
        rms_wav1_rvbt = np.sqrt(np.sum(wav1_rvbt**2)/wav1_length)
    if int(wav2_filtered_duration*sr) == 0:
        #print('wav2_length', wav2_length)
        rms_wav2_rvbt = np.sqrt(np.sum(wav2_rvbt**2)/wav2_length)
    if int(wav1_filtered_duration*sr) != 0:
        rms_wav1_rvbt = np.sqrt(np.sum(wav1_rvbt**2)/int(wav1_filtered_duration*sr))
    if int(wav2_filtered_duration*sr) != 0:
        rms_wav2_rvbt = np.sqrt(np.sum(wav2_rvbt**2)/int(wav2_filtered_duration*sr))
    
    # Convert SIR from dB to a linear scale
    sir_linear = 10**(sir_db / 20)
    
    # Calculate the scaling factor for wav2 to achieve the desired SIR
    scale_factor = rms_wav1_rvbt / (sir_linear * rms_wav2_rvbt)
    
    # Adjust wav2_rvbt by the scaling factor
    adjusted_wav2_rvbt = wav2_rvbt * scale_factor
    
    # Adjust wav2_dp by the scaling factor
    adjusted_wav2_dp = wav2_dp * scale_factor

    return wav1_rvbt, adjusted_wav2_rvbt, wav1_dp, adjusted_wav2_dp

def overlap_wavs(rng, entry1, entry2, wav1, wav1_start_time_original, wav2, wav2_start_time_original, target_duration, sr, wav1_words, wav1_words_end_times, wav2_words, wav2_words_end_times):
    """Adjust two audio files with a random overlap ratio."""

    def filter_words(words, end_times, start_time, end_time):
        filtered_words = [word for word, t in zip(words, end_times) if start_time <= t <= end_time]
        filtered_end_times = [t for t in end_times if start_time <= t <= end_time]
        filtered_duration, voice_duration = calculate_total_duration(filtered_words, filtered_end_times, start_time)
        return " ".join(filtered_words), ",".join(f"{t:.3f}" for t in filtered_end_times), filtered_duration, voice_duration

    def process_with_vad(wav, sr):
        model = load_silero_vad()
        timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True)
        wav_filtered_duration, wav_voice_duration = compute_active_speech_duration(timestamps)
        return wav_filtered_duration, wav_voice_duration, None, None

    wav1_len, wav2_len = len(wav1), len(wav2)

    if wav1_len < 3 * sr or wav2_len < 3 * sr:
        max_len = max(wav1_len, wav2_len)
        min_len = min(wav1_len, wav2_len)
        start_offset = rng.integers(0, max_len - min_len + 1) if max_len > min_len else 0

        wav1_ = np.zeros(max_len)
        wav2_ = np.zeros(max_len)

        if wav1_len == min_len:
            wav1_[start_offset:start_offset + wav1_len] = wav1
            wav2_[:wav2_len] = wav2
        else:
            wav2_[start_offset:start_offset + wav2_len] = wav2
            wav1_[:wav1_len] = wav1

        overlap_len = min(wav1_len, wav2_len)
        overlap_duration = overlap_len / sr

        wav1_end_time_original = wav1_start_time_original + wav1_len / sr
        wav2_end_time_original = wav2_start_time_original + wav2_len / sr

        wav1_start_time_mixture = start_offset / sr if wav1_len == min_len else 0
        wav1_end_time_mixture = (start_offset + wav1_len) / sr if wav1_len == min_len else wav1_len / sr

        wav2_start_time_mixture = start_offset / sr if wav2_len == min_len else 0
        wav2_end_time_mixture = (start_offset + wav2_len) / sr if wav2_len == min_len else wav2_len / sr

    else:
        overlap_len = wav1_len + wav2_len - int(target_duration * sr)
        start_offset = wav1_len - overlap_len
        wav1_ = np.zeros(int(target_duration * sr))
        wav2_ = np.zeros(int(target_duration * sr))
        wav1_[:wav1_len] = wav1
        wav2_[start_offset:start_offset + wav2_len] = wav2
        wav1_start_time_mixture = 0
        wav1_end_time_mixture = wav1_len / sr
        wav2_start_time_mixture = start_offset / sr
        wav2_end_time_mixture = (start_offset + wav2_len) / sr
        wav1_end_time_original = wav1_start_time_original + wav1_len / sr
        wav2_end_time_original = wav2_start_time_original + wav2_len / sr
        overlap_duration = overlap_len / sr

    if wav1_words and wav1_words_end_times:
        wav1_words, wav1_end_times, wav1_filtered_duration, wav1_voice_duration = filter_words(
            wav1_words, wav1_words_end_times, wav1_start_time_original, wav1_end_time_original
        )
    else:
        wav1_filtered_duration, wav1_voice_duration, wav1_words, wav1_end_times = process_with_vad(wav1, sr)

    if wav2_words and wav2_words_end_times:
        wav2_words, wav2_end_times, wav2_filtered_duration, wav2_voice_duration = filter_words(
            wav2_words, wav2_words_end_times, wav2_start_time_original, wav2_end_time_original
        )
    else:
        wav2_filtered_duration, wav2_voice_duration, wav2_words, wav2_end_times = process_with_vad(wav2, sr)

    return (
        wav1_filtered_duration, wav1_voice_duration, wav2_filtered_duration, wav2_voice_duration, overlap_duration,
        wav1_, wav2_, wav1_start_time_mixture, wav1_end_time_mixture, wav2_start_time_mixture, wav2_end_time_mixture,
        wav1_start_time_original, wav1_end_time_original, wav2_start_time_original, wav2_end_time_original,
        wav1_words, wav1_end_times, wav2_words, wav2_end_times
    )


def mix_wavs(wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir, wav1_length, wav2_length, wav1_filtered_duration, wav2_filtered_duration):
    """Mix two wav files with a random overlap ratio and normalize the result."""
    # Apply sir to the target (wav1_rvbt) or interference (wav1_rvbt)
    if wav1_filtered_duration != 0 and wav2_filtered_duration != 0:
        wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp = adjust_wav(wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir, wav1_length, wav2_length, wav1_filtered_duration, wav2_filtered_duration)
        mixed_wav = wav1_rvbt + wav2_rvbt
        # Avoid clipping
        max_scale = np.max([np.max(np.abs(mixed_wav)),np.max(np.abs(wav1_rvbt)),np.max(np.abs(wav2_rvbt)),np.max(np.abs(wav1_dp)),np.max(np.abs(wav2_dp))])
        mixed_wav = mixed_wav / max_scale * 0.9
        wav1_rvbt = wav1_rvbt / max_scale * 0.9
        wav2_rvbt = wav2_rvbt / max_scale * 0.9
        wav1_dp = wav1_dp / max_scale * 0.9
        wav2_dp = wav2_dp / max_scale * 0.9
        return mixed_wav, wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir
    else:
        mixed_wav = wav1_rvbt + wav2_rvbt
        sir = None
        return mixed_wav, wav1_rvbt, wav2_rvbt, wav1_dp, wav2_dp, sir
        
def save_json(data, path):
    """Save JSON data to a file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Error saving JSON to {path}: {e}")
        raise
        
def categorize(value_target, value_inference, categories):
    """ Determine the category based on the given value and classification rules """
    if value_target > value_inference:
        return categories['bigger'], categories['smaller']
    elif value_target < value_inference:
        return categories['smaller'], categories['bigger']
        
def combine_target_and_interference(target_entry, interference_entry, output_dir, mixture_json_filename):
    
    def check_difference(value1, value2, threshold, categories):
        """Classification based on difference and threshold"""
        if abs(value1 - value2) > threshold:
            return categorize(value1, value2, categories)
        else:
            return categories['similar'], categories['similar']

    # Update mean_f0_dp_category
    target_mean_f0_dp = target_entry['mean_f0_dp']
    interference_mean_f0_dp = interference_entry['mean_f0_dp']
    if target_mean_f0_dp is not None and interference_mean_f0_dp is not None:
        target_entry['mean_f0_dp_category'], interference_entry['mean_f0_dp_category'] = check_difference(
            target_mean_f0_dp, interference_mean_f0_dp, 5, {'bigger': 'higher', 'smaller': 'lower', 'similar': 'similar'}
        )
    else:
        target_entry['mean_f0_dp_category'] = None
        interference_entry['mean_f0_dp_category'] = None

    # Update mean_f0_rvbt_category
    target_mean_f0_rvbt = target_entry.get('mean_f0_rvbt')
    interference_mean_f0_rvbt = interference_entry.get('mean_f0_rvbt')
    if target_mean_f0_rvbt is not None and interference_mean_f0_rvbt is not None:
        target_entry['mean_f0_rvbt_category'], interference_entry['mean_f0_rvbt_category'] = check_difference(
            target_mean_f0_rvbt, interference_mean_f0_rvbt, 5, {'bigger': 'higher', 'smaller': 'lower', 'similar': 'similar'}
        )
    else:
        target_entry['mean_f0_rvbt_category'] = None
        interference_entry['mean_f0_rvbt_category'] = None

    # Update log_f0_range_dp_category
    target_log_f0_range_dp = target_entry['log_f0_range_dp']
    interference_log_f0_range_dp = interference_entry['log_f0_range_dp']
    if target_log_f0_range_dp is not None and interference_log_f0_range_dp is not None:
        target_entry['log_f0_range_dp_category'], interference_entry['log_f0_range_dp_category'] = check_difference(
            target_log_f0_range_dp, interference_log_f0_range_dp, 0.097, # pitch range 差异20%, logf0 差异 0.079; pitch range 差异25%, logf0 差异 0.097
            {'bigger': 'wider', 'smaller': 'narrower', 'similar': 'similar'}
        )
    else:
        target_entry['log_f0_range_dp_category'] = None
        interference_entry['log_f0_range_dp_category'] = None

    # Update log_f0_range_rvbt_category
    target_log_f0_range_rvbt = target_entry['log_f0_range_rvbt']
    interference_log_f0_range_rvbt = interference_entry['log_f0_range_rvbt']
    if target_log_f0_range_rvbt is not None and interference_log_f0_range_rvbt is not None:
        target_entry['log_f0_range_rvbt_category'], interference_entry['log_f0_range_rvbt_category'] = check_difference(
            target_log_f0_range_rvbt, interference_log_f0_range_rvbt, 0.097, 
            {'bigger': 'wider', 'smaller': 'narrower', 'similar': 'similar'}
        )
    else:
        target_entry['log_f0_range_rvbt_category'] = None
        interference_entry['log_f0_range_rvbt_category'] = None

    # Update speaking_rate_category
    target_speaking_rate = target_entry['speaking_rate']
    interference_speaking_rate = interference_entry['speaking_rate']
    if target_speaking_rate is not None and interference_speaking_rate is not None:
        target_entry['speaking_rate_category'], interference_entry['speaking_rate_category'] = check_difference(
            target_speaking_rate, interference_speaking_rate, 0.15*min(target_speaking_rate, interference_speaking_rate), 
            {'bigger': 'faster', 'smaller': 'slower', 'similar': 'similar'}
        )
    else:
        target_entry['speaking_rate_category'] = None
        interference_entry['speaking_rate_category'] = None

    # Update actual_duration_category
    target_actual_duration = target_entry['actual_duration']
    interference_actual_duration = interference_entry['actual_duration']
    if target_actual_duration is not None and interference_actual_duration is not None:
        target_entry['actual_duration_category'], interference_entry['actual_duration_category'] = check_difference(
            target_actual_duration, interference_actual_duration, 0.15*min(target_actual_duration, interference_actual_duration), 
            {'bigger': 'longer', 'smaller': 'shorter', 'similar': 'similar'}
        )
    else:
        target_entry['actual_duration_category'] = None
        interference_entry['actual_duration_category'] = None
    
    # Update speaking_duration_category
    target_actual_duration_words = target_entry['filtered_duration']
    interference_actual_duration_words = interference_entry['filtered_duration']
    if target_actual_duration_words is not None and interference_actual_duration_words is not None:
        target_entry['speaking_duration_category'], interference_entry['speaking_duration_category'] = check_difference(
            target_actual_duration_words, interference_actual_duration_words, 0.15*min(target_actual_duration_words, interference_actual_duration_words), 
            {'bigger': 'longer', 'smaller': 'shorter', 'similar': 'similar'}
        )
    else:
        target_entry['speaking_duration_category'] = None
        interference_entry['speaking_duration_category'] = None

    # Update loundness_category
    target_loundness = target_entry['sir']
    interference_loundness = interference_entry['sir']
    if target_loundness is not None and interference_loundness is not None:
        target_entry['loundness_category'], interference_entry['loundness_category'] = check_difference(
            target_loundness, interference_loundness, 3, 
            {'bigger': 'louder', 'smaller': 'quieter', 'similar': 'similar'}
        )
    else:
        target_entry['loundness_category'] = None
        interference_entry['loundness_category'] = None

    # Update distance_category
    target_distance = target_entry['distance']
    interference_distance = interference_entry['distance']
    if target_distance is not None and interference_distance is not None:
        target_entry['distance_category'], interference_entry['distance_category'] = check_difference(
            target_distance, interference_distance, 0.5, 
            {'bigger': 'farther', 'smaller': 'nearer', 'similar': 'similar'}
        )
    else:
        target_entry['distance_category'] = None
        interference_entry['distance_category'] = None

    # Update age_category
    target_age = target_entry.get('age')
    interference_age = interference_entry.get('age')
    if target_age is not None and interference_age is not None:
        target_entry['age_category'], interference_entry['age_category'] = check_difference(
            target_age, interference_age, 10, 
            {'bigger': 'older', 'smaller': 'younger', 'similar': 'similar'}
        )
    else:
        target_entry['age_category'] = None
        interference_entry['age_category'] = None

    return target_entry, interference_entry

def calculate_f0_features(audio, sr, voice_duration):

    if not np.isfinite(audio).all():
        print("Audio contains non-finite values (NaN or Inf).")
        return None, None

    try:
        # Use pyin to extract f0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, sr=sr, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=512,
            win_length=256, hop_length=128
        )

        # filter out Nan values (voiced_flag = True)
        f0 = f0[~np.isnan(f0)]

        # filter out zero values
        f0_nonzero = f0[f0 > 0]

        if len(f0_nonzero) > 0:
            mean_f0 = round(np.mean(f0_nonzero), 2)
            if np.max(f0_nonzero) == np.min(f0_nonzero):
                log_f0_range = None
            else:
                log_f0_range = round(np.log10(np.max(f0_nonzero) - np.min(f0_nonzero)), 2)
            return mean_f0, log_f0_range
        else:
            print("No voiced segments detected.")
            return None, None

    except Exception as e:
        print("Error during f0 calculation:", e)
        return None, None

# Function to remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Count syllables based on language
def count_syllables(text, language):
    if text is None:  # Check if text is None
        return 0

    text = remove_punctuation(text)

    if language == 'chinese':  # Chinese: Each character is a syllable
        return len(text)
    elif language == 'english':  # English syllable count
        return sum([len(re.findall(r'[aeiouy]+', word.lower())) for word in text.split()])
    elif language == 'french':  # French syllable count
        return sum([len(re.findall(r'[aeiouy]+', word.lower())) for word in text.split()])
    elif language == 'german':  # German syllable count
        return sum([len(re.findall(r'[aeiouäöü]+', word.lower())) for word in text.split()])
    elif language == 'spanish':  # Spanish syllable count
        return sum([len(re.findall(r'[aeiouáéíóúü]+', word.lower())) for word in text.split()])
    else:
        return 0

# Calculate speaking speed based on syllables and duration
def calculate_speaking_rate(text, duration, language):
    if text is None:  # Check if text is None and return None
        return None

    syllable_count = count_syllables(text, language)
    if syllable_count > 0 and duration > 0:
        return syllable_count / duration * 60
    else:
        return None

# Format words and end_times to the specified format
def format_word_alignment(words, end_times):
    word_string = ",".join(words)
    time_string = ",".join((end_times))
    
    return [word_string, time_string]

def random_wav(target_entry, rng):
    """
      1. 找到目标文件的第一层目录 root_dir_path。
      2. 用 pathlib.Path.rglob() 遍历 root_dir_path 的所有子孙目录，
         判断路径部件中是否含有 speaker_code 这一级。如果有，就记录下来。
      3. 从记录的这些目录里随机选一个，在其下 (含子目录) 找 .wav/.WAV 文件，
         排除与 target_file_basename (不区分大小写) 相同的文件。
      4. 如果没找到，则在 target_file 同级目录下，前缀= speaker_code 的 .wav/.WAV 文件中再随机选一个。
      5. 都找不到返回 None。
    """

    # -------------------------
    # 1. 解析基本信息
    # -------------------------
    target_file = target_entry['file_path']
    target_dir = os.path.dirname(target_file)             
    target_file_basename = os.path.basename(target_file).lower()

    speaker_id = target_entry.get('speaker_id', '')
    if '_' in speaker_id:
        speaker_code = speaker_id.split('_', 1)[1]  
    else:
        speaker_code = speaker_id

    # -------------------------
    # 2. 获取“第一层文件夹” root_dir_name
    # -------------------------
    stripped_path = target_dir.strip(os.sep)
    parts = stripped_path.split(os.sep)
    if not parts:
        root_dir_name = ''
    else:
        root_dir_name = parts[0]

    if not root_dir_name:
        root_dir_path = Path(target_dir)
    else:
        # 不使用 os.getcwd()，直接用 root_dir_name 作为相对路径
        root_dir_path = Path(root_dir_name)

    # -------------------------------------------------
    # 3. 在 root_dir_path 下，用 rglob() 找到可能含有 speaker_code 的目录
    #    （不使用 os.walk，而是 pathlib 的递归方式）
    # -------------------------------------------------
    speaker_subdir_paths = set()
    # rglob("*") 会遍历 root_dir_path 以及所有子孙目录/文件
    for p in root_dir_path.rglob("*"):
        if p.is_dir():
            # p.parts 是一个元组，如 ('chinese','ESD','0004','Sad')
            if speaker_code in p.parts:
                # 找到 speaker_code 所在层级
                idx = p.parts.index(speaker_code)
                # 拼回到该层级为止
                subdir_path = Path(*p.parts[:idx+1])
                speaker_subdir_paths.add(subdir_path.resolve())

    speaker_subdir_paths = list(speaker_subdir_paths)
    if speaker_subdir_paths:
        chosen_speaker_subdir = rng.choice(speaker_subdir_paths)
        # 4. 在 chosen_speaker_subdir 下递归找所有后缀为 .wav/.WAV 的文件
        all_wav_files = []
        for wav_path in chosen_speaker_subdir.rglob("*"):
            if wav_path.is_file():
                # 后缀判断：suffix 如 ".wav", ".WAV", 统一小写后比较
                if wav_path.suffix.lower() == ".wav":
                    # 排除与 target_file_basename 同名(大小写不敏感)
                    if wav_path.name.lower() != target_file_basename:
                        all_wav_files.append(str(wav_path))

        if all_wav_files:
            chosen_wav = rng.choice(all_wav_files)
            return chosen_wav

    # -------------------------
    # 如果没找到可用目录或目录下没文件，
    # 就在 target_file 同级目录下，找前缀= speaker_code 的 .wav/.WAV
    # -------------------------
    candidate_wavs = []
    for f in os.listdir(target_dir):
        if (f.lower().endswith('.wav')
            and f.lower() != target_file_basename
            and f.startswith(speaker_code)):
            candidate_wavs.append(os.path.join(target_dir, f))

    if candidate_wavs:
        chosen = rng.choice(candidate_wavs)
        return chosen


    return None
