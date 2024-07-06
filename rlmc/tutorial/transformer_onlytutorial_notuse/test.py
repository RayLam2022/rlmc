# -*- coding:utf-8 -*-
# @Time    : 2023/1/13 23:20


from datasets import *

def test(model, enc_input, start_symbol):
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

enc_inputs, dec_inputs, dec_outputs = make_data()
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(loader))
model = torch.load('model.pth')

# predict_dec_input = test(model, enc_inputs[0].view(1, -1).cuda(), start_symbol=tgt_vocab["S"])
# predict, _, _, _ = model(enc_inputs[0].view(1, -1).cuda(), predict_dec_input)
# predict=predict.data.max(1, keepdim=True)[1]


predict_dec_input=['S','P','P','P','P']
predict_dec_input = torch.tensor([[tgt_vocab[symbol] for symbol in predict_dec_input]]).T.cuda()
for i in range(tgt_len):
    predict, _, _, _ = model(enc_inputs[0].view(1, -1).cuda(), predict_dec_input.T)
    
    pre=predict.data.max(1, keepdim=True)[1][i][0]

    if pre != tgt_vocab["E"] and i < tgt_len-2:
        predict_dec_input[i+1][0] = pre
        predict_dec_input[i+2:][0] = tgt_vocab["P"]
    elif pre != tgt_vocab["E"] and i < tgt_len-1:
        predict_dec_input[i+1][0] = pre
    elif pre == tgt_vocab["E"] and i < tgt_len-1:
        predict_dec_input[i+1:][0]= tgt_vocab["E"]
        break


predict=predict_dec_input[1:].cpu()
element_to_add = torch.tensor([[tgt_vocab['E']]])
predict = torch.cat((predict, element_to_add), dim=0)

print([src_idx2word[int(i)] for i in enc_inputs[0]], '->',
      [idx2word[n.item()] for n in predict.squeeze()])