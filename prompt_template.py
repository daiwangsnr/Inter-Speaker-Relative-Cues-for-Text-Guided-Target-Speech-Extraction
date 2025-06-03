from itertools import chain, combinations
import json

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

# Use handcrafted prompts

def build_style_description_from(style, keys, rng):
    description = 'the'
    
    if 'age_category' in keys:
        age_category = style['age_category']
        description += ' '
        description += age_category

    if 'gender' in keys:
        description += ' '
        #description += 'female' if style['gender'] == 'F' else 'male'
        description += style['gender']
    
    if 'language' in keys:
        description += ' ' + style['language'].capitalize() + '-speaking'

    description += ' speaker'

    
    if 'aligned_transcription' in keys:
        description += ' who says: "' + style['aligned_transcription'] + '",'
    
    if 'speaking_duration_category' in keys:
        if 'aligned_transcription' in keys:
            description += (' has a ' + style['speaking_duration_category'] + ' speaking duration').replace('\\', '')
        else:
            description += (' who has a ' + style['speaking_duration_category'] + ' speaking duration').replace('\\', '')

    if 'emotion' in keys or 'mean_f0_rvbt_category' in keys or 'log_f0_range_rvbt_category' in keys or 'speaking_rate_category' in keys or 'loundness_category' in keys:
        description += rng.choice([' characterized by a', ' with a'])


    keys1 = [key for key in keys if key in ['emotion', 'mean_f0_rvbt_category', 'log_f0_range_rvbt_category', 'speaking_rate_category', 'loundness_category']]

    for i, key in enumerate(keys1):
        if i == 0:
            description += ' '
        elif i == len(keys1) - 1:
            if len(keys1) > 2:
                description += ', and '
            else:
                description += ' and '
        else:
            description += ', '
        
        

        description += style[key]

        description += ' '
        if key == 'emotion':
            description += 'emotion'
        if key == 'mean_f0_rvbt_category':
            description += 'pitch level'
        if key == 'log_f0_range_rvbt_category':
            description += 'pitch range'
        if key == 'speaking_rate_category':
            description += 'speaking speed'
        if key == 'loundness_category':
            description += 'loudness'
       
    if 'distance_category' in keys:
        description += ', and stands ' + style['distance_category'] + ' to the microphone'

    if 'temporal_order' in keys:
        description += ', and appears ' + style['temporal_order']

    description += rng.choice([' in the audio', ' in the speech mixture']) 
    

    return description


def build_short_style_descriptions_for_two_speakers(target_entry, interference_entry):
    '''
    Describe speakers only by the style difference.
    Let (x1, y1, z1) and (x2, y2, z2) be the style differece tuples
    if they differ in three of five attributes. For example, x is gender, 
    y is energy, z is emotion. We then choose nonempty subsets from tuples
    e.g. (x1, z1) and (z2). Finally, we generate prompts from (x1, z1) and (z2). 
    The goal is generate more human-like prompt, because we may not describe the 
    speakers in all aspects, but only in some aspects in which speakers differ.
    '''

    s1 = (target_entry['aligned_transcription'], target_entry['language'], target_entry['gender'], target_entry['emotion'], target_entry['temporal_order'], target_entry['age_category'], target_entry['speaking_rate_category'], target_entry['speaking_duration_category'], target_entry['mean_f0_rvbt_category'], target_entry['log_f0_range_rvbt_category'], target_entry['loundness_category'], target_entry['distance_category'])
    s2 = (interference_entry['aligned_transcription'], interference_entry['language'], interference_entry['gender'], interference_entry['emotion'], interference_entry['temporal_order'], interference_entry['age_category'], interference_entry['speaking_rate_category'], interference_entry['speaking_duration_category'], interference_entry['mean_f0_rvbt_category'], interference_entry['log_f0_range_rvbt_category'], interference_entry['loundness_category'], interference_entry['distance_category'])

    # Two speakers must be distinguishable in the mixture
    assert s1 != s2

    keys = ['aligned_transcription', 'language', 'gender', 'emotion', 'temporal_order', 'age_category', 'speaking_rate_category', 'speaking_duration_category', 'mean_f0_rvbt_category', 'log_f0_range_rvbt_category', 'loundness_category', 'distance_category']
    keys_d = [] # all attributes they differ
    s1_d = {} # all attributes and values spk1 differs from spk2
    s2_d = {} # all attributes and values spk2 differs from spk1
    for key, v1, v2 in zip(keys, s1, s2):
        if v1 != v2:
            s1_d[key] = v1
            s2_d[key] = v2
            keys_d.append(key)

    # Select a subset of all attributes they differ 
    keys_d = random.choice(powerset(keys_d))

    description1 = build_style_description_from(s1_d, keys_d, rng)
    description2 = build_style_description_from(s2_d, keys_d, rng)

    return description1, description2

class ShortTemplate():

    def __init__(
        self,
        acts=['1'],
        random=True,
        rng='rng',
    ):
        #acts.sort()
        assert acts == ['1']        
        self.lookups = {
            '0': ['remove {}', 'eliminate {}', 'take {} away'],
            '1': ['extract {}', 'pick out {}', 'isolate {}'],
        }

        self.lookups_extract_or_remove = {
            '0': ['remove', 'eliminate', 'take away {}'],
            '1': ['extract', 'pick out', 'isolate'],
        }

        self.templates = {
            1: [
                'Please {}.',
                'I want to {}.',
                'Can you {}?'
            ],
            2: [
                'Please {} and {}.',
                'I want to {} and {}.',
                'Can you {} and {}?'
            ],
            3: [
                'Please {}, {}, and {}.',
                'I want to {}, {}, and {}.',
                'Can you {}, {}, and {}?'
            ],
            4: [
                'Please {}, {}, {}, and {}.',
                'I want to {}, {}, {}, and {}.',
                'Can you {}, {}, {}, and {}?'
            ]
        }

        self.templates_extract_or_remove = {
            1: [
                'Please {} {}.',
                'I want to {} {}.',
                'Can you {} {}?'
            ],
            2: [
                'Please {} {} and {}.',
                'I want to {} {} and {}.',
                'Can you {} {} and {}?'
            ],
            3: [
                'Please {} {}, {}, and {}.',
                'I want to {} {}, {}, and {}.',
                'Can you {} {}, {}, and {}?'
            ],
        }

        self.shuffle = False
        self.random = True

        print(f'Initialized {str(self.__class__.__name__)}: ')
        #print(f'shuffle: {str(shuffle)} random: {str(random)}')


# target_style, interference_style = build_short_style_descriptions_for_two_speakers(target_entry, interference_entry)
# acts = ['1']
# spks = [target_style]
# prompt = ShortTemplate(acts, spks=spks) 
