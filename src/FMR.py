import json
import os
from ClipContextual import clip
from PIL import Image
import pickle
import torch
import torch.nn.functional as F
from typing import List

def compose_discrete_prompts(
    tokenizer,
    process_entities: List[str],
) -> torch.Tensor:

    prompt_head = 'There are'
    prompt_tail = ' in image.'

    if len(process_entities) == 0: # without entities
        discrete_prompt =  prompt_head + ' something' + prompt_tail
    else:
        discrete_prompt = ''
        for entity in process_entities: # gpt2 in transformer encoder ' ' + word into one token by default
            discrete_prompt += ' ' + entity + ','     # ' person, dog, park,'
        discrete_prompt = discrete_prompt[:-1]        # ' person, dog, park'
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail # 'There are person, dog, park in image.'

    # entities_tokens = torch.tensor(tokenizer.encode(discrete_prompt))   # (discrete_prompt_length, ) 
    entities_tokens = torch.tensor(tokenizer(discrete_prompt)).clone().detach()   # (discrete_prompt_length, ) 

    return entities_tokens

def cosine_similarity(vec1, vec2):
    # 确保 vec1 和 vec2 的最后一维是相同i    assert vec1.size(-1) == vec2.size(-1), "Last dimension of input tensors must match."
    if len(vec1.shape) < 3:
        vec1 = vec1.unsqueeze(0)
    
    # vec2 = vec2.expand(1, seq1, -1)  # [bs1, seq2, dim]

    # 计算点积
    dot_product = torch.bmm(vec1, vec2.transpose(1, 2))  # [bs, seq1, seq2]

    # 计算范数
    norm_vec1 = vec1.norm(dim=2, keepdim=True)  # [bs, seq, 1]
    norm_vec2 = vec2.norm(dim=2, keepdim=True)  # [bs, seq, 1]

    # 计算余弦相似度
    similarity = dot_product / (norm_vec1 * norm_vec2.transpose(1, 2))

    # 计算平均相似度
    average_similarity = similarity.mean(dim=[1, 2])  # [bs]

    return average_similarity

def find_similar_indices(tensor_a, tensor_b, threshold):
    # 计算余弦相似度
    sim = F.cosine_similarity(tensor_a, tensor_b, dim=-1)  # 在最后一维计算相似度
    
    # 获取大于阈值的索引
    indices = (sim > threshold).nonzero(as_tuple=True)[1]
    
    return indices

@torch.no_grad()
def main(datasets, encoder, proprecess, annotations, outpath, alpha):

    results = []

    if datasets == 'coco' or datasets == 'flickr30k': # coco, flickr30k
        # format = {image_path: [caption1, caption2, ...]} -> [[image_path, image_features, [caption1, caption2, ...]], ...]
        if datasets == 'coco':
            rootpath = 'coco_img_path'
        elif datasets == 'flickr30k':
            rootpath = 'flickr30k_img_path'
        with open("detected_entity_path", "r") as file:
            data_entities = json.load(file)
        for image_id in annotations:
            caption = annotations[image_id]
            image_path = rootpath + image_id
            image = Image.open(image_path).convert('RGB')
            image = proprecess(image).unsqueeze(0).half().to("cuda:0")
            detected_objects = list(filter(lambda x: x[0] >= 5, data_entities[image_id]))
            prompt_tokens = [l[1] for l in detected_objects]
            prompt_tokens = compose_discrete_prompts(clip.tokenize, prompt_tokens).to("cuda:0")
            
            with torch.no_grad():
                text_cls, _, _ = encoder.encode_text(prompt_tokens)
                text_cls /= text_cls.norm(dim=-1, keepdim=True)
                text_cls = text_cls.unsqueeze(0)
                bs = image.size(0)
                cls_features, img_context, img_proj, attn_weight = encoder.visual(image.half().to("cuda:0"), require_attn=True)

                cls_features /= cls_features.norm(dim=-1, keepdim=True)
                clip_conx = img_context @ img_proj
                clip_conx /= clip_conx.norm(dim=-1, keepdim=True)

                top_cls_patch_ids = find_similar_indices(clip_conx, text_cls, alpha).unsqueeze(0)
                mixed_patch_feature = []
                for idx in range(bs):
                    tp_idx = top_cls_patch_ids[idx]
                    top_weight = attn_weight[idx, tp_idx].softmax(dim=-1)
                    top_features = top_weight @ clip_conx[idx]
                    mixed_patch_feature.append(top_features.unsqueeze(0))

                mixed_patch_feature = torch.cat(mixed_patch_feature, dim=0)
                mixed_patch_feature = F.normalize(mixed_patch_feature, dim=-1)
                FMR = cls_features.unsqueeze(1) + mixed_patch_feature
                FMR = FMR.squeeze(0)
            results.append([image_id, FMR, caption])

    else: # nocaps
        # format = [{'split': 'near_domain', 'image_id': '4499.jpg', 'caption': [caption1, caption2, ...]}, ...]
        # format = [[image_path, image_split, image_features, [caption1, captions2, ...]], ...]
        rootpath = 'nocaps_img_path'
        with open("annotations/retrieved_entity/caption_coco_image_nocaps_7.json", "r") as file:
            data_entities = json.load(file)
        for annotation in annotations:
            split = annotation['split']
            image_id = annotation['image_id']
            caption = annotation['caption']
            image_path = rootpath + image_id
            image = Image.open(image_path).convert('RGB')
            image = proprecess(image).unsqueeze(0).half()
            detected_objects = list(filter(lambda x: x[0] >= 5, data_entities[image_id]))
            prompt_tokens = [l[1] for l in detected_objects]
            prompt_tokens = compose_discrete_prompts(clip.tokenize, prompt_tokens).to("cuda:0")
            with torch.no_grad():
                text_cls, _, _ = encoder.encode_text(prompt_tokens)
                text_cls /= text_cls.norm(dim=-1, keepdim=True)
                text_cls = text_cls.unsqueeze(0)
                bs = image.size(0)
                cls_features, img_context, img_proj, attn_weight = encoder.visual(image.half().to("cuda:0"), require_attn=True)

                cls_features /= cls_features.norm(dim=-1, keepdim=True)
                clip_conx = img_context @ img_proj
                clip_conx /= clip_conx.norm(dim=-1, keepdim=True)

                top_cls_patch_ids = find_similar_indices(clip_conx, text_cls, alpha).unsqueeze(0)
                mixed_patch_feature = []
                for idx in range(bs):
                    tp_idx = top_cls_patch_ids[idx]
                    top_weight = attn_weight[idx, tp_idx].softmax(dim=-1)
                    top_features = top_weight @ clip_conx[idx]
                    mixed_patch_feature.append(top_features.unsqueeze(0))

                mixed_patch_feature = torch.cat(mixed_patch_feature, dim=0)
                mixed_patch_feature = F.normalize(mixed_patch_feature, dim=-1)
                if mixed_patch_feature.shape[1] != 0:
                    FMR = cls_features.unsqueeze(1) + mixed_patch_feature
                    FMR = FMR.squeeze(0)
                else:
                    FMR = cls_features
            results.append([image_id, split, FMR, caption])

    with open(outpath, 'wb') as outfile:
        pickle.dump(results, outfile)

if __name__ == '__main__':
    
    device = 'cuda:0'
    clip_type = 'ViT-B/32'
    clip_name = clip_type.replace('/', '')
    alpha = 0.3
    
    path_nocaps = './annotations/nocaps/nocaps_corpus.json'
    path_val_coco = './annotations/coco/val_captions.json'
    path_test_coco = './annotations/coco/test_captions.json'
    path_val_flickr30k = './annotations/flickr30k/val_captions.json'
    path_test_flickr30k = './annotations/flickr30k/test_captions.json'

    outpath_nocaps = f'./annotations/nocaps/nocaps_corpus_{clip_name}_{alpha}.pickle'
    outpath_val_coco = f'./annotations/coco/val_captions_{clip_name}_{alpha}.pickle'
    outpath_test_coco = f'./annotations/coco/coco_test_captions_{clip_name}_{alpha}.pickle'
    outpath_val_flickr30k = f'./annotations/flickr30k/val_captions_{clip_name}_{alpha}.pickle'
    outpath_test_flickr30k = f'./annotations/flickr30k/flickr30k_test_captions_{clip_name}_{alpha}.pickle'

    # format = [{'split': 'near_domain', 'image_id': '4499.jpg', 'caption': [caption1, caption2, ...]}, ...]
    # format = [[image_path, image_split, image_features, [caption1, captions2, ...]], ...]
    with open(path_nocaps, 'r') as infile:
        nocaps = json.load(infile)
    
    # format = {image_path: [caption1, caption2, ...]} -> [[image_path, image_features, [caption1, caption2, ...]], ...]
    with open(path_val_coco, 'r') as infile:
        val_coco = json.load(infile)
    with open(path_test_coco, 'r') as infile:
        test_coco = json.load(infile)
    
    with open(path_val_flickr30k, 'r') as infile:
        val_flickr30k = json.load(infile)
    
    with open(path_test_flickr30k, 'r') as infile:
        test_flickr30k = json.load(infile)

    encoder, proprecess = clip.load(clip_type, device)

    if not os.path.exists(outpath_nocaps):
        main('nocaps', encoder, proprecess, nocaps, outpath_nocaps, alpha)

    if not os.path.exists(outpath_val_coco):
        main('coco', encoder, proprecess, val_coco, outpath_val_coco, alpha)

    if not os.path.exists(outpath_test_coco):
        main('coco', encoder, proprecess, test_coco, outpath_test_coco, alpha)

    if not os.path.exists(outpath_val_flickr30k):
        main('flickr30k', encoder, proprecess, val_flickr30k, outpath_val_flickr30k, alpha)

    if not os.path.exists(outpath_test_flickr30k):
        main('flickr30k', encoder, proprecess, test_flickr30k, outpath_test_flickr30k, alpha)