
import torch

#TODO: implement option to include the conditioning bit of input in the output
def autoregressive_generation_multimodal(features, model, autoreg_mods=[], teacher_forcing=False, ground_truth=False):
    inputs_ = []
    for i,mod in enumerate(model.input_mods):
        input_ = features["in_"+mod]
        if model.input_types[i] == "c":
            input_ = torch.from_numpy(input_).float().to(model.device)
        else:
            input_ = torch.from_numpy(input_).long().to(model.device)
        inputs_.append(input_)
    output_time_offsets = model.output_time_offsets
    input_time_offsets = model.input_time_offsets
    input_lengths = model.input_lengths
    output_lengths = model.output_lengths
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
    #sequence_length = inputs_[0].shape[0]
    #TODO: make this less ad-hoc
    sequence_length = inputs_[-1].shape[0]
    print(sequence_length)
    #import pdb;pdb.set_trace()
    with torch.no_grad():
        # for t in range(min(512, sequence_length-max(input_lengths)-1)):
        import time
        start_time = time.time()
        for t in range(sequence_length-max(input_lengths)+1):
        #for t in range(512):
            print(t)
            inputs = [x.clone().to(model.device) for x in input_tmp]
            # import pdb;pdb.set_trace()

            if not ground_truth:
                outputs = model.forward(inputs)

            #outputs[0][:,0,-4] = 0.0
            #outputs[0][:,0,-6] = 0.0
            if t == 0:
                for i, mod in enumerate(output_mods):
                    # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)

                    if not ground_truth:
                        output = outputs[i]
                    else:
                        j = input_mods.index(mod)
                        output = inputs_[j][t+output_time_offsets[i]+output_lengths[i]:t+output_time_offsets[i]+output_lengths[i]+1]
                    output_seq.append(output[:1].detach().clone())

                    #output_seq.append(inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]+0.15*torch.randn(1,219).to(model.device))
            else:
                for i, mod in enumerate(output_mods):
                    #output_seq[i] = torch.cat([output_seq[i], inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]+0.15*torch.randn(1,219).to(model.device)])

                    if not ground_truth:
                        output = outputs[i]
                    else:
                        j = input_mods.index(mod)
                        output = inputs_[j][t+output_time_offsets[i]+output_lengths[i]:t+output_time_offsets[i]+output_lengths[i]+1]
                    output_seq[i] = torch.cat([output_seq[i], output[:1].detach().clone()])

                    # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                    # print(outputs[i][:1])
            if t < sequence_length-1:
                for i, mod in enumerate(input_mods):
                    if mod in autoreg_mods:
                        j = output_mods.index(mod)
                        if not ground_truth:
                            output = outputs[j]
                        else:
                            output = inputs_[i][t+input_time_offsets[j]+input_lengths[j]:t+input_time_offsets[j]+input_lengths[j]+1]
                        if teacher_forcing:
                            input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][t+input_time_offsets[i]+input_lengths[i]:t+input_time_offsets[i]+input_lengths[i]+1]],0)
                        else:
                            # import pdb;pdb.set_trace()
                            input_tmp[i] = torch.cat([input_tmp[i][1:],output[:1].detach().clone()],0)
                        # print(torch.mean((inputs_[i][t+input_time_offsets[i]+input_lengths[i]+1:t+input_time_offsets[i]+input_lengths[i]+1+1]-outputs[j][:1].detach().clone())**2))

                        if not ground_truth:
                            print(torch.mean((inputs_[i][t+output_time_offsets[j]:t+output_time_offsets[j]+1]-outputs[j][:1].detach().clone())**2))
                    else:
                        if model.input_fix_length_types[i] == "single":
                            #input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][input_time_offsets[i]+input_lengths[i]+t:input_time_offsets[i]+input_lengths[i]+t+1]],0)
                            pass
                        else:
                            input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][input_time_offsets[i]+input_lengths[i]+t:input_time_offsets[i]+input_lengths[i]+t+1]],0)

    print("--- %s seconds ---" % (time.time() - start_time))
    return output_seq

