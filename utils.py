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


def generate_sentc(model, sp, n_seq, prompt, device):
    enc_ids = sp.encode_as_ids(prompt)
    if len(enc_ids) > n_seq:
        print("입력하신 문장이 너무 깁니다. 다시 입력해주세요.")
        return
    enc_ids += [0] * (n_seq - len(enc_ids))
    # 초기 디코더 입력 설정 (BOS 토큰으로 시작)
    dec_ids = [sp.bos_id()] + [0] * (n_seq - 1)
    
    # Tensor로 변환하고 디바이스에 할당
    enc_ids = torch.tensor(enc_ids).unsqueeze(0).to(device)
    dec_ids = torch.tensor(dec_ids).unsqueeze(0).to(device)
    
    # Transformer 디코더 루프
    for i in range(n_seq - 1):
        output = model(enc_ids, dec_ids)
        
        # 다음 토큰 예측
        pred = output[0, i].argmax().item()
        
        # EOS 토큰을 예측한 경우 중단
        if pred == sp.eos_id():
            break
        
        # 다음 디코더 입력 업데이트
        dec_ids[0, i + 1] = pred
    # 최종 출력 문장 디코딩
    generated_ids = dec_ids[0].tolist()
    generated_sentence = sp.DecodeIds(generated_ids)

    return generated_sentence

        