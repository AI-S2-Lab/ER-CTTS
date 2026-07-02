import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D, pad_2d_2,pad_1d_2, pad_sequences

from string import punctuation as strpunc

from transformers import XLNetConfig, XLNetLMHeadModel, XLNetModel, XLNetTokenizer

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]

        self.history_fine_type = model_config["history_encoder"]["fine_type"]  

        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]

        self.text_emb_tod_size = model_config["history_encoder"]["text_emb_tod_size"]  

        
        self.waviemocap_emb_size = model_config["history_encoder"]["waviemocap_emb_size"]  

        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        
        self.probability_bool = preprocess_config["preprocessing"]["probability_bool"]

        
        self.probability_words_path = preprocess_config["preprocessing"]["probability_words_path"]
        self.probability_words_path_unsigned = preprocess_config["preprocessing"]["probability_words_path_unsigned"]

        
        self.model_path = model_config["path"]["xlnet_model_path"]

        
        self.tokenizer = XLNetTokenizer.from_pretrained(self.model_path)

        
        self.max_len = model_config["emphasis_predictor"]["max_len"]

        self.pitch_level_tag, self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)

        self.basename, self.speaker, self.text, self.raw_text, self.emotion, self.emphasis_probability = self.process_meta(
            filename
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        
        with open(self.probability_words_path) as f:
            self.emphasis_probability_words = json.load(f)

        with open(self.probability_words_path_unsigned) as f:
            self.emphasis_probability_unsigned_words = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]

        
        words_probability = self.emphasis_probability_words[basename]["probability"]
        raw_text_by_json = self.emphasis_probability_words[basename]["words"]

        
        
        
        emphasis_tuple = self.emphasis_data(raw_text_by_json, words_probability)


        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel_{}".format(self.pitch_level_tag),
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        if self.learn_alignment:
            attn_prior_path = os.path.join(
                self.preprocessed_path,
                "attn_prior",
                "{}-attn_prior-{}.npy".format(speaker, basename),
            )
            attn_prior = np.load(attn_prior_path)
            duration = None
        else:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            attn_prior = None
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)

        
        history_text = list()
        history_text_emb = list()
        history_text_len = list()

        
        history_waviemocap_emb = list()
        history_waviemocap_len = list()

        
        history_text_words_tod_emb = list()
        history_text_words_tod_len = list()
        history_text_words_tod_mask = list()

        
        history_audio_fine_emb = list()

        
        history_words_probability = list()
        history_words_probability_len = list()

        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None

        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]

            
            if self.history_fine_type == "Seven":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)

                
                waviemocap_emb_path = os.path.join(
                    self.preprocessed_path,
                    "waviemocap_emb",
                    "{}-waviemocap_emb-{}.npy".format(speaker, basename),
                )
                
                waviemocap_emb = np.load(waviemocap_emb_path)

                
                text_words_tod_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_words_tod_emb",
                    "{}-text_words_tod_emb-{}.npy".format(speaker, basename),
                )
                
                text_words_tod_emb = np.load(text_words_tod_emb_path)
                
                text_words_tod_emb_len = np.shape(text_words_tod_emb)[0]
                
                text_words_tod_emb = pad_2d_2(text_words_tod_emb, self.max_len)

                
                audio_fine_emb_path = os.path.join(
                    self.preprocessed_path,
                    "audio_fine_emb",
                    "{}-audio_fine_emb-{}.npy".format(speaker, basename),
                )
                
                audio_fine_emb = np.load(audio_fine_emb_path)[0]
                
                audio_fine_emb_len = np.shape(audio_fine_emb)[0]
                
                

                for i, h_basename in enumerate(history_basenames):
                    
                    h_speaker = h_basename.split("_")[1]
                    
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    
                    history_text_emb.append(h_text_emb)

                    
                    h_waviemocap_emb_path = os.path.join(
                        self.preprocessed_path,
                        "waviemocap_emb",
                        "{}-waviemocap_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    
                    h_waviemocap_emb = np.load(h_waviemocap_emb_path)
                    
                    history_waviemocap_emb.append(h_waviemocap_emb)

                    
                    h_text_words_tod_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_words_tod_emb",
                        "{}-text_words_tod_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    
                    h_text_words_tod_emb = np.load(h_text_words_tod_emb_path)

                    
                    history_text_words_tod_len.append(np.shape(h_text_words_tod_emb)[0])

                    
                    h_text_words_tod_emb = pad_2d_2(h_text_words_tod_emb, self.max_len)
                    
                    history_text_words_tod_emb.append(h_text_words_tod_emb)

                    
                    h_audio_fine_emb_path = os.path.join(
                        self.preprocessed_path,
                        "audio_fine_emb",
                        "{}-audio_fine_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    
                    h_audio_fine_emb = np.load(h_audio_fine_emb_path)[0]
                    
                    history_audio_fine_emb.append(h_audio_fine_emb)

                    history_speaker.append(h_speaker_id)
                    
                    h_words_probability = self.emphasis_probability_unsigned_words[h_basename]["probability"].split()
                    h_words_probability_len = len(h_words_probability)

                    history_words_probability_len.append(h_words_probability_len)
                    
                    h_words_probability = pad_1d_2(h_words_probability, self.max_len)
                    history_words_probability.append(h_words_probability)
                    
                    if i == history_len - 1 and history_len < self.max_history_len:
                        

                        self.pad_history(
                            self.max_history_len - history_len,
                            history_text_emb=history_text_emb,
                            history_waviemocap_emb=history_waviemocap_emb,
                            history_speaker=history_speaker,
                            history_text_words_tod_emb=history_text_words_tod_emb,
                            history_words_probability=history_words_probability,
                        )

                if turn == 0:
                    
                    self.pad_history(  
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_waviemocap_emb=history_waviemocap_emb,
                        history_speaker=history_speaker,
                        history_text_words_tod_emb=history_text_words_tod_emb,
                        history_words_probability=history_words_probability,
                    )

                
                history = {
                    "text_emb": text_emb,  
                    
                    "text_words_tod_emb": text_words_tod_emb,  
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,  
                    "history_waviemocap_emb": history_waviemocap_emb,  
                    "history_speaker": history_speaker,
                    "history_text_words_tod_emb": history_text_words_tod_emb,  
                    "history_text_words_tod_len": history_text_words_tod_len,  
                    
                    "history_audio_fine_emb": history_audio_fine_emb,  
                    "history_words_probability": history_words_probability,  
                    "history_words_probability_len": history_words_probability_len,  
                }

                    
                sample = {
                    "id": basename,
                    "speaker": speaker_id,
                    "text": phone,
                    "raw_text": raw_text,
                    "mel": mel,
                    "pitch": pitch,
                    "energy": energy,
                    "duration": duration,
                    "attn_prior": attn_prior,
                    "spker_embed": spker_embed,
                    "emotion": emotion_id,
                    "text_words_tod_emb_len": text_words_tod_emb_len,  
                    "history": history,
                    "emphasis_tuple": emphasis_tuple,  
                }

        return sample

    
    def emphasis_data(self, raw_text, words_probability):
        
        raw_text_list = raw_text.split()
        words_probability_list = words_probability.split()

        
        xlnet_tokens = []
        
        xlnet_tokens.append("<s>")
        
        orig_to_tok_map = []
        
        
        labels = []
        
        words_len = 0
        
        
        
        
        

        
        punctuation_mask = []

        
        if (len(raw_text_list) > len(words_probability_list)):
            raw_text_list = raw_text_list[:len(words_probability_list)]

        
        for i in range(len(words_probability_list)):
            
            
            
            
            
            
            orig_to_tok_map.append(len(xlnet_tokens))
            
            labels.append(words_probability_list[i])
            
            xlnet_tokens.extend(self.tokenizer.tokenize(raw_text_list[i]))

            
            if raw_text_list[i] in strpunc: 
                punctuation_mask.append(0)
            else:
                punctuation_mask.append(1)

        
        words_len = len(raw_text_list)
        
        xlnet_tokens.append("</s>")

        
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in xlnet_tokens]

        
        return (
            input_ids,  
            orig_to_tok_map,  
            labels,  
            words_len,  
            punctuation_mask, 
        )

    def pad_history(self,
                    pad_size,
                    history_text=None,
                    history_text_emb=None,
                    history_waviemocap_emb=None,
                    history_text_len=None,
                    history_waviemocap_len=None,
                    history_pitch=None,
                    history_energy=None,
                    history_duration=None,
                    history_emotion=None,
                    history_speaker=None,
                    history_mel_len=None,
                    history_text_words_tod_emb=None,
                    history_text_words_tod_len=None,
                    history_text_words_tod_mask=None,
                    history_words_probability=None,
        ):
        for _ in range(pad_size):  
            history_text.insert(0, np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.insert(0, np.zeros(self.text_emb_size,
                                                dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.insert(0,
                                    0) if history_text_len is not None else None  

            
            history_waviemocap_emb.insert(0, np.zeros(self.waviemocap_emb_size,
                                                      dtype=np.float32)) if history_waviemocap_emb is not None else None
            history_waviemocap_len.insert(0,
                                          0) if history_waviemocap_len is not None else None  

            
            history_text_words_tod_emb.insert(0, np.zeros((self.max_len, self.text_emb_tod_size),
                                                          dtype=np.float32)) if history_text_words_tod_emb is not None else None
            history_text_words_tod_len.insert(0, 0) if history_text_words_tod_len is not None else None

            history_pitch.insert(0, np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.insert(0, np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.insert(0, np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.insert(0,
                                   0) if history_emotion is not None else None  
            history_speaker.insert(0,
                                   0) if history_speaker is not None else None  
            history_mel_len.insert(0,
                                   0) if history_mel_len is not None else None  

            
            history_words_probability.insert(0,
                                             np.zeros(self.max_len,
                                                      dtype=np.float32)) if history_words_probability is not None else None

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []

            
            emphasis_probability = []

            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e, zy  = line.strip("\n").split("|")
                else:
                    n, s, t, r, e, zy  = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)

                
                if self.probability_bool:
                    
                    emphasis_probability.append(zy.split())

            return name, speaker, text, raw_text, emotion, emphasis_probability


    
    def get_attention_masks(self, input_shape):
        masks_datas = []
        for seq in input_shape:
            seq_mask = [float(i > 0) for i in seq]
            masks_datas.append(seq_mask)

        return masks_datas

    
    def pad_punc_mask(self, punc_mask_batch):
        
        max = 0
        for item in punc_mask_batch:
            if len(item)>max:
                max = len(item)

        
        for i in range(len(punc_mask_batch)):
            for j in range(max - len(punc_mask_batch[i])):
                
                punc_mask_batch[i].append(0)


        return  punc_mask_batch


    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs] if not self.learn_alignment else None
        attn_priors = [data[idx]["attn_prior"] for idx in idxs] if self.learn_alignment else None
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([data[idx]["emotion"] for idx in idxs]) if self.load_emotion else None

        
        
        input_ids_batch = [data[idx]["emphasis_tuple"][0] for idx in idxs]  
        token_map_batch = [data[idx]["emphasis_tuple"][1] for idx in idxs]  
        token_label_batch = [data[idx]["emphasis_tuple"][2] for idx in idxs]  
        words_len_batch = [data[idx]["emphasis_tuple"][3] for idx in idxs]  
        punc_mask_batch = [data[idx]["emphasis_tuple"][4] for idx in idxs]  

        
        text_words_tod_emb_lens = [data[idx]["text_words_tod_emb_len"] for idx in idxs]

        
        

        input_ids_batch = pad_sequences(input_ids_batch, maxlen=self.max_len, dtype="long", truncating="post",
                                        padding="post")  
        token_map_batch = pad_sequences(token_map_batch, maxlen=self.max_len, dtype="long", truncating="post",
                                        padding="post")  
        token_label_batch = pad_sequences(token_label_batch, maxlen=self.max_len, dtype="float", truncating="post",
                                          padding="post")  

        
        input_ids_masks = self.get_attention_masks(input_ids_batch)  

        
        punc_mask_batch = self.pad_punc_mask(punc_mask_batch)
        
        emphasis_data_batch = (
            input_ids_batch,  
            token_map_batch,  
            token_label_batch,  
            words_len_batch,  
            input_ids_masks,  
            punc_mask_batch,  
        )

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        if self.learn_alignment:
            attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        else:
            durations = pad_1D(durations)

        history_info = None
        if self.history_fine_type != "none":
            if self.history_fine_type == "Seven":
                
                text_embs = [data[idx]["history"]["text_emb"] for idx in idxs]
                history_lens = [data[idx]["history"]["history_len"] for idx in idxs]
                history_text_embs = [data[idx]["history"]["history_text_emb"] for idx in idxs]
                history_speakers = [data[idx]["history"]["history_speaker"] for idx in idxs]

                
                
                history_waviemocap_embs = [data[idx]["history"]["history_waviemocap_emb"] for idx in idxs]

                
                text_words_tod_embs = [data[idx]["history"]["text_words_tod_emb"] for idx in idxs]
                history_text_words_tod_embs = [data[idx]["history"]["history_text_words_tod_emb"] for idx in idxs]
                history_text_words_tod_len = [data[idx]["history"]["history_text_words_tod_len"] for idx in idxs]

                
                
                history_audio_fine_embs = [data[idx]["history"]["history_audio_fine_emb"] for idx in idxs]

                
                history_words_probabilitys = [data[idx]["history"]["history_words_probability"] for idx in idxs]
                history_words_probability_lens = [data[idx]["history"]["history_words_probability_len"] for idx in idxs]

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)
                
                
                history_waviemocap_embs = np.array(history_waviemocap_embs)

                
                text_words_tod_embs = np.array(text_words_tod_embs)
                history_text_words_tod_embs = np.array(history_text_words_tod_embs)

                
                
                

                
                history_words_probabilitys = np.array(history_words_probabilitys)

                
                history_info = (
                    text_embs,  
                    
                    text_words_tod_embs,  
                    history_lens,
                    history_text_embs,  
                    history_waviemocap_embs,  
                    history_text_words_tod_embs,  
                    history_speakers,
                    history_text_words_tod_len,  
                    
                    history_audio_fine_embs,  
                    history_words_probabilitys,  
                    history_words_probability_lens,  
                )

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            history_info,
            emphasis_data_batch,  
            text_words_tod_emb_lens,  
        )

    def collate_fn(self, data):
        
        data_size = len(data)  

        
        if self.sort:
            
            
            len_arr = np.array([d["text"].shape[0] for d in data])

            
            idx_arr = np.argsort(-len_arr)
        else:
            
            idx_arr = np.arange(data_size)

        
        
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]

        
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()

        
        if not self.drop_last and len(tail) > 0:
            
            idx_arr += [tail.tolist()]

        
        output = list()

        
        for idx in idx_arr:
            
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.dataset_name = preprocess_config["dataset"]
        self.history_fine_type = model_config["history_encoder"]["fine_type"]  
        self.text_emb_tod_size = model_config["history_encoder"]["text_emb_tod_size"]  
        
        self.waviemocap_emb_size = model_config["history_encoder"]["waviemocap_emb_size"]  
        
        self.probability_bool = preprocess_config["preprocessing"]["probability_bool"]

        
        self.probability_words_path = preprocess_config["preprocessing"]["probability_words_path"]
        self.probability_words_path_unsigned = preprocess_config["preprocessing"]["probability_words_path_unsigned"]
        
        self.model_path = model_config["path"]["xlnet_model_path"]

        
        self.tokenizer = XLNetTokenizer.from_pretrained(self.model_path)

        
        self.max_len = model_config["emphasis_predictor"]["max_len"]




        self.basename, self.speaker, self.text, self.raw_text, self.emotion, self.emphasis_probability = self.process_meta(
            filepath
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)

        
        with open(self.probability_words_path) as f:
            self.emphasis_probability_words = json.load(f)

        with open(self.probability_words_path_unsigned) as f:
            self.emphasis_probability_unsigned_words = json.load(f)



    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]

        
        words_probability = self.emphasis_probability_words[basename]["probability"]
        raw_text_by_json = self.emphasis_probability_words[basename]["words"]

        
        
        
        emphasis_tuple = self.emphasis_data(raw_text_by_json, words_probability)


        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None


        
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)

        
        history_text = list()
        history_text_emb = list()
        history_text_len = list()

        
        history_waviemocap_emb = list()
        history_waviemocap_len = list()

        
        history_text_words_tod_emb = list()
        history_text_words_tod_len = list()
        history_text_words_tod_mask = list()

        
        history_audio_fine_emb = list()

        
        history_words_probability = list()
        history_words_probability_len = list()

        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None
        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]

            
            if self.history_fine_type == "Seven":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)

                
                
                
                
                
                
                
                

                
                text_words_tod_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_words_tod_emb",
                    "{}-text_words_tod_emb-{}.npy".format(speaker, basename),
                )
                
                text_words_tod_emb = np.load(text_words_tod_emb_path)
                
                text_words_tod_emb_len = np.shape(text_words_tod_emb)[0]
                
                text_words_tod_emb = pad_2d_2(text_words_tod_emb, self.max_len)

                
                
                
                
                
                
                
                
                
                
                
                

                for i, h_basename in enumerate(history_basenames):
                    
                    h_speaker = h_basename.split("_")[1]
                    
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    
                    history_text_emb.append(h_text_emb)

                    
                    h_waviemocap_emb_path = os.path.join(
                        self.preprocessed_path,
                        "waviemocap_emb",
                        "{}-waviemocap_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    
                    h_waviemocap_emb = np.load(h_waviemocap_emb_path)
                    
                    history_waviemocap_emb.append(h_waviemocap_emb)

                    
                    h_text_words_tod_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_words_tod_emb",
                        "{}-text_words_tod_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    
                    h_text_words_tod_emb = np.load(h_text_words_tod_emb_path)

                    
                    history_text_words_tod_len.append(np.shape(h_text_words_tod_emb)[0])

                    
                    h_text_words_tod_emb = pad_2d_2(h_text_words_tod_emb, self.max_len)
                    
                    history_text_words_tod_emb.append(h_text_words_tod_emb)

                    
                    h_audio_fine_emb_path = os.path.join(
                        self.preprocessed_path,
                        "audio_fine_emb",
                        "{}-audio_fine_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    
                    h_audio_fine_emb = np.load(h_audio_fine_emb_path)[0]
                    
                    history_audio_fine_emb.append(h_audio_fine_emb)

                    history_speaker.append(h_speaker_id)
                    
                    h_words_probability = self.emphasis_probability_unsigned_words[h_basename]["probability"].split()
                    h_words_probability_len = len(h_words_probability)

                    history_words_probability_len.append(h_words_probability_len)
                    
                    h_words_probability = pad_1d_2(h_words_probability, self.max_len)
                    history_words_probability.append(h_words_probability)
                    
                    if i == history_len - 1 and history_len < self.max_history_len:
                        

                        self.pad_history(
                            self.max_history_len - history_len,
                            history_text_emb=history_text_emb,
                            history_waviemocap_emb=history_waviemocap_emb,
                            history_speaker=history_speaker,
                            history_text_words_tod_emb=history_text_words_tod_emb,
                            history_words_probability=history_words_probability,
                        )

                if turn == 0:
                    
                    self.pad_history(  
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_waviemocap_emb=history_waviemocap_emb,
                        history_speaker=history_speaker,
                        history_text_words_tod_emb=history_text_words_tod_emb,
                        history_words_probability=history_words_probability,
                    )
                history = {
                    "text_emb": text_emb,  
                    
                    "text_words_tod_emb": text_words_tod_emb,  
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,  
                    "history_waviemocap_emb": history_waviemocap_emb,  
                    "history_speaker": history_speaker,
                    "history_text_words_tod_emb": history_text_words_tod_emb,  
                    "history_text_words_tod_len": history_text_words_tod_len,  
                    
                    "history_audio_fine_emb": history_audio_fine_emb,  
                    "history_words_probability": history_words_probability,  
                    "history_words_probability_len": history_words_probability_len,  
                }

        return (basename, speaker_id, phone, raw_text, spker_embed, emotion_id, history, text_words_tod_emb_len, emphasis_tuple)

    
    def emphasis_data(self, raw_text, words_probability):
        
        raw_text_list = raw_text.split()
        words_probability_list = words_probability.split()

        
        xlnet_tokens = []
        
        xlnet_tokens.append("<s>")
        
        orig_to_tok_map = []
        
        
        labels = []
        
        words_len = 0
        
        
        
        
        

        
        punctuation_mask = []

        
        if (len(raw_text_list) > len(words_probability_list)):
            raw_text_list = raw_text_list[:len(words_probability_list)]

        
        for i in range(len(words_probability_list)):
            
            
            
            
            
            
            orig_to_tok_map.append(len(xlnet_tokens))
            
            labels.append(words_probability_list[i])
            
            xlnet_tokens.extend(self.tokenizer.tokenize(raw_text_list[i]))

            
            if raw_text_list[i] in strpunc: 
                punctuation_mask.append(0)
            else:
                punctuation_mask.append(1)

        
        words_len = len(raw_text_list)
        
        xlnet_tokens.append("</s>")

        
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in xlnet_tokens]

        
        return (
            input_ids,  
            orig_to_tok_map,  
            labels,  
            words_len,  
            punctuation_mask, 
        )


    def pad_history(self,
                    pad_size,
                    history_text=None,
                    history_text_emb=None,
                    history_waviemocap_emb=None,
                    history_text_len=None,
                    history_waviemocap_len=None,
                    history_pitch=None,
                    history_energy=None,
                    history_duration=None,
                    history_emotion=None,
                    history_speaker=None,
                    history_mel_len=None,
                    history_text_words_tod_emb=None,
                    history_text_words_tod_len=None,
                    history_text_words_tod_mask=None,
                    history_words_probability=None,
                    ):
        for _ in range(pad_size):  
            history_text.insert(0, np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.insert(0, np.zeros(self.text_emb_size,
                                                dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.insert(0,
                                    0) if history_text_len is not None else None  

            
            history_waviemocap_emb.insert(0, np.zeros(self.waviemocap_emb_size,
                                                      dtype=np.float32)) if history_waviemocap_emb is not None else None
            history_waviemocap_len.insert(0,
                                          0) if history_waviemocap_len is not None else None  

            
            history_text_words_tod_emb.insert(0, np.zeros((self.max_len, self.text_emb_tod_size),
                                                          dtype=np.float32)) if history_text_words_tod_emb is not None else None
            history_text_words_tod_len.insert(0, 0) if history_text_words_tod_len is not None else None

            history_pitch.insert(0, np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.insert(0, np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.insert(0, np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.insert(0,
                                   0) if history_emotion is not None else None  
            history_speaker.insert(0,
                                   0) if history_speaker is not None else None  
            history_mel_len.insert(0,
                                   0) if history_mel_len is not None else None  

            
            history_words_probability.insert(0,
                                             np.zeros(self.max_len,
                                                      dtype=np.float32)) if history_words_probability is not None else None

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []


            
            emphasis_probability = []

            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e, zy = line.strip("\n").split("|")
                else:
                    n, s, t, r, e, zy = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)

                
                if self.probability_bool:
                    
                    emphasis_probability.append(zy.split())

            return name, speaker, text, raw_text, emotion, emphasis_probability




    
    def get_attention_masks(self, input_shape):
        masks_datas = []
        for seq in input_shape:
            seq_mask = [float(i > 0) for i in seq]
            masks_datas.append(seq_mask)

        return masks_datas

    
    def pad_punc_mask(self, punc_mask_batch):
        
        max = 0
        for item in punc_mask_batch:
            if len(item)>max:
                max = len(item)

        
        for i in range(len(punc_mask_batch)):
            for j in range(max - len(punc_mask_batch[i])):
                
                punc_mask_batch[i].append(0)


        return  punc_mask_batch
    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([d[5] for d in data]) if self.load_emotion else None

        
        
        input_ids_batch = [d[-1][0] for d in data]  
        token_map_batch = [d[-1][1] for d in data]   
        token_label_batch = [d[-1][2] for d in data]   
        words_len_batch = [d[-1][3] for d in data]  
        punc_mask_batch = [d[-1][4] for d in data]   

        
        text_words_tod_emb_lens = [d[-2] for d in data]

        
        

        input_ids_batch = pad_sequences(input_ids_batch, maxlen=self.max_len, dtype="long", truncating="post",
                                        padding="post")  
        token_map_batch = pad_sequences(token_map_batch, maxlen=self.max_len, dtype="long", truncating="post",
                                        padding="post")  
        token_label_batch = pad_sequences(token_label_batch, maxlen=self.max_len, dtype="float", truncating="post",
                                          padding="post")  

        
        input_ids_masks = self.get_attention_masks(input_ids_batch)  

        
        punc_mask_batch = self.pad_punc_mask(punc_mask_batch)
        
        emphasis_data_batch = (
            input_ids_batch,  
            token_map_batch,  
            token_label_batch,  
            words_len_batch,  
            input_ids_masks,  
            punc_mask_batch,  
        )
        texts = pad_1D(texts)

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [d[6]["text_emb"] for d in data]
                history_lens = [d[6]["history_len"] for d in data]
                history_text_embs = [d[6]["history_text_emb"] for d in data]
                history_speakers = [d[6]["history_speaker"] for d in data]



                
                
                history_waviemocap_embs = [d[6]["history_waviemocap_emb"] for d in data]

                
                text_words_tod_embs = [d[6]["text_words_tod_emb"] for d in data]
                history_text_words_tod_embs = [d[6]["history_text_words_tod_emb"] for d in data]
                history_text_words_tod_len = [d[6]["history_text_words_tod_len"] for d in data]

                
                
                history_audio_fine_embs = [d[6]["history_audio_fine_emb"] for d in data]

                
                history_words_probabilitys = [d[6]["history_words_probability"] for d in data]
                history_words_probability_lens = [d[6]["history_words_probability_len"] for d in data]

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)
                
                
                history_waviemocap_embs = np.array(history_waviemocap_embs)

                
                text_words_tod_embs = np.array(text_words_tod_embs)
                history_text_words_tod_embs = np.array(history_text_words_tod_embs)

                
                
                

                
                history_words_probabilitys = np.array(history_words_probabilitys)


                
                history_info = (
                    text_embs,  
                    
                    text_words_tod_embs,  
                    history_lens,
                    history_text_embs,  
                    history_waviemocap_embs,  
                    history_text_words_tod_embs,  
                    history_speakers,
                    history_text_words_tod_len,  
                    
                    history_audio_fine_embs,  
                    history_words_probabilitys,  
                    history_words_probability_lens,  
                )

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds, emotions, history_info, emphasis_data_batch, text_words_tod_emb_lens
