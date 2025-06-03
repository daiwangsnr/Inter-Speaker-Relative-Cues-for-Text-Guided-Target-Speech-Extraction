import os
import json
import random
import librosa
import soundfile as sf
import numpy as np
import gpuRIR # Ensure you have gpuRIR installed and set up correctly.

# Initialize random number generators
rngs = [random.Random(i) for i in range(28, 78, 7)]

# Function to calculate the Euclidean distance
def calculate_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

# Generate random source position
def generate_source_position(rng9, rng10, rng11, room_sz, pos_mic):
    while True:
        source_x = rng9.uniform(0, room_sz[0])
        source_y = rng10.uniform(0, room_sz[1])
        source_position_xy = [source_x, source_y]
        pos_mic_xy = pos_mic[:2]
        distance = round(calculate_distance(source_position_xy, pos_mic_xy), 2)
        if (0.3 <= distance <= 1.5):
            return [source_position_xy[0], source_position_xy[1], rng11.uniform(1.6, 1.9)], distance

def generate_rir(room_sz, rt60, pos_mic, pos_src, fs=16000):
    pos_mic = np.array([pos_mic])
    pos_src = np.array([pos_src])
    att_max = 60.0 # Attenuation at the end of the simulation [dB]
    beta = gpuRIR.beta_SabineEstimation(room_sz, rt60)
    Tdiff= None # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n(Tmax, room_sz)	# Number of image sources in each dimension
    RIR = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=pos_src, pos_rcv=pos_mic, nb_img=nb_img, Tmax=Tmax, Tdiff=Tdiff, fs=fs)
    #nb_img = rir.t2n(rt60, room_sz )	# Number of image sources in each dimension
    #RIR = rir.simulateRIR(room_sz, beta, source_pos, pos_mic, nb_img, rt60, fs)
    RIR = RIR.squeeze(0).squeeze(0)
    return RIR
    

def generate_rir_dp(room_sz, rt60, pos_mic, pos_src, fs=16000):
    pos_mic = np.array([pos_mic])
    pos_src = np.array([pos_src])
    """Generate the room impulse response using gpuRIR."""
    beta = [0]*6
    Tmax = 0.1 # Time to stop the simulation [s]
    nb_img = [1,1,1]# Number of image sources in each dimension
    Tdiff = None
    RIR = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=pos_src, pos_rcv=pos_mic, nb_img=nb_img, Tmax=Tmax, Tdiff=Tdiff, fs=fs)
    RIR = RIR.squeeze(0).squeeze(0)
    return RIR

# Create directories for train, dev, and test
for dataset in ['rirs_distance/train', 'rirs_distance/dev', 'rirs_distance/test']:
    os.makedirs(dataset, exist_ok=True)

# Generate rir data
rir_datasets = {'rirs/train': 10000, 'rirs/dev': 1000, 'rirs/test': 1000}
for rir_dataset, count in rir_datasets.items():
    for i in range(count):
        room_sz = [rngs[1].uniform(9, 11), rngs[2].uniform(9, 11), rngs[3].uniform(2.6, 3.5)]
        print('room_sz:', room_sz)
        rt60 = round(rngs[4].uniform(0.3, 0.6),3)
        pos_mic = [room_sz[0] / 2, room_sz[1] / 2, 1.5]

        pos1_source, distance1 = generate_source_position(rngs[5], rngs[6], rngs[7], room_sz, pos_mic)
        pos2_source, distance2 = generate_source_position(rngs[5], rngs[6], rngs[7], room_sz, pos_mic)

        rir1 = generate_rir(room_sz, rt60, pos_mic, pos1_source)
        rir1_dp = generate_rir_dp(room_sz, 0, pos_mic, pos1_source)
        rir2 = generate_rir(room_sz, rt60, pos_mic, pos2_source)
        rir2_dp = generate_rir_dp(room_sz, 0, pos_mic, pos2_source)
        rir_distance = str(distance1)+'_'+str(distance2)
        # Save RIRs with naming convention
        np.savez(os.path.join(rir_dataset, f'{i}_{rt60}_{rir_distance}.npz'),
                 rir1=rir1, rir1_dp=rir1_dp, rir2=rir2, rir2_dp=rir2_dp,
                 )


