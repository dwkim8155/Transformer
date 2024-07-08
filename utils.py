import torch
import numpy as np

def position_encoding(n_seq, n_embd):
    pos_table = []
    for pos in range(n_seq):
        pos_table.append([pos/np.power(10000, (2*(i_embd//2))/n_embd) for i_embd in range(n_embd)])
    pos_table = np.array(pos_table)
    pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])
    pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])
    pos_table = torch.FloatTensor(pos_table)
    return pos_table


def making_padding_mask(attention_score, token_ids):
        padding_pos = token_ids == 0
        attention_score = attention_score.masked_fill(padding_pos.unsqueeze(1) , float('-inf'))
        return attention_score


def shift_right(tensor, bos_token_id):
    bos_token = tensor.new_full((tensor.size(0), 1), bos_token_id)
    return torch.cat([bos_token, tensor[:, :-1]], dim=-1)


import torch

def generate_sentc(model, sp, n_seq, prompt, device):
    enc_ids = sp.encode_as_ids(prompt)
    if len(enc_ids) > n_seq:
        print("입력하신 문장이 너무 깁니다. 다시 입력해주세요.")
        return

    enc_ids += [0] * (n_seq - len(enc_ids))  # 패딩 추가
    dec_ids = [sp.bos_id()] + [0] * (n_seq - 1)  # 초기 디코더 입력 설정

    enc_ids = torch.tensor(enc_ids).unsqueeze(0).to(device)
    dec_ids = torch.tensor(dec_ids).unsqueeze(0).to(device)
    soft_max = torch.nn.Softmax(dim=-1).to(device)  # dim=-1로 수정
    
    for i in range(n_seq - 1):
        output = model(enc_ids, dec_ids)
        
        # 다음 토큰 예측
        output = soft_max(output[:, -1, :])  # 마지막 시퀀스 위치에 대해서 softmax 적용
        pred = output.argmax(dim=-1).item()
        
        if pred == sp.eos_id():
            return sp.decode_ids(dec_ids[0].tolist())
        
        # 다음 디코더 입력 업데이트
        dec_ids[0, i + 1] = pred

    final_output = model(enc_ids, dec_ids)
    final_output = soft_max(final_output[:, -1, :])
    pred = final_output.argmax(dim=-1).item()
    
    #dec_ids에서 BOS 제거하고 마지막 pred 추가
    
    generated_id = dec_ids[0].tolist()
    generated_id.pop(0)
    generated_id.append(pred)
    generated_sentence = sp.decode_ids(generated_id)
    
    return generated_sentence
