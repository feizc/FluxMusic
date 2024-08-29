import os 
import json 

def test_reconstuct(): 
    import yaml 
    from diffusers import AutoencoderKL
    from transformers import SpeechT5HifiGan
    from audioldm2.utilities.data.dataset import AudioDataset
    from utils import load_clip, load_clap, load_t5

    model_path = '/maindata/data/shared/multimodal/public/dataset_music/audioldm2' 
    config = yaml.load(
        open(
            'config/16k_64.yaml',
            'r'
        ),
        Loader=yaml.FullLoader,
    )
    print(config)
    t5 = load_t5('cuda', max_length=256)
    clap = load_clap('cuda', max_length=256)

    dataset = AudioDataset(
        config=config, split="train", waveform_only=False, dataset_json_path='mini_dataset.json',
        tokenizer=clap.tokenizer, 
        uncond_pro=0.1,
        text_ctx_len=77,
        tokenizer_t5=t5.tokenizer,
        text_ctx_len_t5=256,
        uncond_pro_t5=0.1, 
    )
    print(dataset[0]['log_mel_spec'].unsqueeze(0).unsqueeze(0).size()) 

    vae = AutoencoderKL.from_pretrained(os.path.join(model_path, 'vae'))
    vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(model_path, 'vocoder')) 
    latents = vae.encode(dataset[0]['log_mel_spec'].unsqueeze(0).unsqueeze(0)).latent_dist.sample().mul_(vae.config.scaling_factor)
    print('laten size:', latents.size()) 
 
    latents = 1 / vae.config.scaling_factor * latents
    mel_spectrogram = vae.decode(latents).sample
    print(mel_spectrogram.size())
    if mel_spectrogram.dim() == 4:
        mel_spectrogram = mel_spectrogram.squeeze(1)
    waveform = vocoder(mel_spectrogram)
    waveform = waveform[0].cpu().float().detach().numpy()
    print(waveform.shape)
    # import soundfile as sf
    # sf.write('reconstruct.wav', waveform, samplerate=16000) 
    from  scipy.io import wavfile 
    # wavfile.write('reconstruct.wav', 16000, waveform) 



def mini_dataset(num=32): 
    data = []
    for i in range(num):
        data.append(
            {
                'wav': 'case.mp3',
                'label': 'a beautiful music',
            }
        )

    with open('mini_dataset.json', 'w') as f:
        json.dump(data, f, indent=4)


def fma_dataset(): 
    import pandas as pd 

    annotation_prex = "/maindata/data/shared/public/zhengcong.fei/dataset/dataset_music/annotation"
    annotation_list = ['test-00000-of-00001.parquet',  'train-00000-of-00001.parquet',  'valid-00000-of-00001.parquet']
    dataset_prex = '/maindata/data/shared/public/zhengcong.fei/dataset/dataset_music/fma_large'

    data = []
    for annotation_file in annotation_list:
        annotation_file = os.path.join(annotation_prex, annotation_file) 
        df=pd.read_parquet(annotation_file) 
        print(df.shape) 
        for id, row in df.iterrows(): 
            #print(id, row['pseudo_caption'], row['path'])
            tmp_path = os.path.join(dataset_prex, row['path'] + '.mp3')
            # print(tmp_path)
            if os.path.exists(tmp_path): 
                data.append(
                    {
                        'wav': tmp_path,
                        'label': row['pseudo_caption'],
                    }
                )
            # break 
    print(len(data))
    with open('fma_dataset.json', 'w') as f:
        json.dump(data, f, indent=4)





def audioset_dataset(): 
    import pandas as pd  
    dataset_prex = '/maindata/data/shared/public/zhengcong.fei/dataset/dataset_music/audioset' 
    annotation_path = '/maindata/data/shared/public/zhengcong.fei/dataset/dataset_music/audioset/balanced_train-00000-of-00001.parquet'
    df=pd.read_parquet(annotation_path) 
    print(df.shape) 

    data = []
    for id, row in df.iterrows(): 
        #print(id, row['pseudo_caption'], row['path'])
        try:
            tmp_path = os.path.join(dataset_prex, row['path'] + '.flac')
        except: 
            print(row['path'])
        
        if os.path.exists(tmp_path): 
            # print(tmp_path)
            data.append(
                {
                    'wav': tmp_path,
                    'label': row['pseudo_caption'],
                }
            )
    print(len(data))
    with open('audioset_dataset.json', 'w') as f:
        json.dump(data, f, indent=4)
            


def combine_dataset(): 
    data_list = ['fma_dataset.json', 'audioset_dataset.json'] 

    data = []
    for data_file in data_list: 
        with open(data_file, 'r') as f: 
            data += json.load(f)
    print(len(data))
    with open('combine_dataset.json', 'w') as f:
        json.dump(data, f, indent=4)



def test_music_format(): 
    import torchaudio 
    filename = '2.flac'
    waveform, sr = torchaudio.load(filename,) 
    print(waveform, sr )


def test_flops():
    version = 'giant' 
    import torch
    from constants import build_model
    from thop import profile
    
    model = build_model(version).cuda() 
    img_ids = torch.randn((1, 1024, 3)).cuda()
    txt = torch.randn((1, 256, 4096)).cuda()
    txt_ids = torch.randn((1, 256, 3)).cuda()
    y = torch.randn((1, 768)).cuda()
    x = torch.randn((1, 1024, 32)).cuda()
    t = torch.tensor([1] * 1).cuda()
    flops, _ = profile(model, inputs=(x, img_ids, txt, txt_ids, t, y,)) 
    print('FLOPs = ' + str(flops * 2/1000**3) + 'G')


# test_music_format()
# test_reconstuct() 
# mini_dataset()
# fma_dataset() 
# audioset_dataset() 
# combine_dataset() 
test_flops()