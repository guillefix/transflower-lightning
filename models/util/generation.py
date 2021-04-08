
import torch

def autoregressive_generation_multimodal(features, model, autoreg_mods=[], teacher_forcing=False):
    inputs_ = []
    for i,mod in enumerate(model.input_mods):
        input_ = features["in_"+mod]
        input_ = torch.from_numpy(input_).float().to(model.device)
        inputs_.append(input_)
    input_time_offsets = model.input_time_offsets
    input_lengths = model.input_lengths
    input_mods = model.input_mods
    output_mods = model.output_mods
    # predicted_inputs = model.predicted_inputs
    for mod in autoreg_mods:
        assert mod in output_mods

    input_tmp = []
    for i,mod in enumerate(input_mods):
        input_tmp.append(inputs_[i].clone()[input_time_offsets[i]:input_time_offsets[i]+input_lengths[i]])

    #TODO: append the initial conditioning bit to the output too
    model.eval()
    output_seq = []
    # sequence_length = inputs_[0].shape[0]
    sequence_length = inputs_[1].shape[0]
    print(sequence_length)
    with torch.no_grad():
        # for t in range(min(512, sequence_length-max(input_lengths)-1)):
        for t in range(sequence_length-max(input_lengths)-1):
            print(t)
            inputs = [x.clone().cuda() for x in input_tmp]
            outputs = model.forward(inputs)
            if t == 0:
                for i, mod in enumerate(output_mods):
                    output = outputs[i]
                    # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                    output_seq.append(output[:1].detach().clone())
                    # output_seq.append(inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]+0.15*torch.randn(1,219).cuda())
            else:
                for i, mod in enumerate(output_mods):
                    # output_seq[i] = torch.cat([output_seq[i], inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]+0.15*torch.randn(1,219).cuda()])
                    output = outputs[i]
                    output_seq[i] = torch.cat([output_seq[i], output[:1].detach().clone()])
                    # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                    # print(outputs[i][:1])
            if t < sequence_length-1:
                for i, mod in enumerate(input_mods):
                    if mod in autoreg_mods:
                        j = output_mods.index(mod)
                        output = outputs[i]
                        if teacher_forcing:
                            input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]],0)
                        else:
                            # import pdb;pdb.set_trace()
                            input_tmp[i] = torch.cat([input_tmp[i][1:],output[:1].detach().clone()],0)
                        # print(torch.mean((inputs_[i][t+input_time_offsets[i]+input_lengths[i]-predicted_inputs[i]+1:t+input_time_offsets[i]+input_lengths[i]-predicted_inputs[i]+1+1]-outputs[j][:1].detach().clone())**2))
                        print(torch.mean((inputs_[i][t+input_time_offsets[i]+input_lengths[i]+1:t+input_time_offsets[i]+input_lengths[i]+1+1]-outputs[j][:1].detach().clone())**2))
                    else:
                        input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][input_time_offsets[i]+input_lengths[i]+t:input_time_offsets[i]+input_lengths[i]+t+1]],0)

    return output_seq

