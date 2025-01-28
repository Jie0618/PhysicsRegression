import numpy as np
import torch
import torch.nn as nn


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ModelWrapper(nn.Module):
    """"""

    def __init__(
        self,
        env=None,
        embedder=None,
        encoder=None,
        decoder=None,
        beam_type="search",
        beam_length_penalty=1,
        beam_size=1,
        beam_early_stopping=True,
        max_generated_output_len=200,
        beam_temperature=1.0,
    ):
        super().__init__()

        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        self.device = next(self.embedder.parameters()).device

    @torch.no_grad()
    def forward(
        self, input, hints
    ):

        """
        x: bags of sequences (B, T)
        """

        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder


        #TODO:!!!
        #at now, we require units as hints! cant ignore it!!!
        #cause we need it for after process
        units = hints[0]

        B, T = len(input), max([len(xi) for xi in input])
        outputs = [[], [], []]

        for chunk in chunks(
            np.arange(B),
            max(1, 
                min(
                    int(10000 / T),
                    int(100000 / self.beam_size / self.max_generated_output_len),
                ),
            )
        ):
            
            hint = [
                [_hints[idx] for idx in chunk] for _hints in hints
            ]

            x, x_len = embedder([input[idx] for idx in chunk], hint)
            encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)
            bs = encoded.shape[0]

            #y_unit are useless for inference and may cause problem
            #cause some are unknown
            unit = [units[idx] for idx in chunk]
            x_unit = [[env.equation_encoder.units_encode(u) for u in un[:-1]] for un in unit]
            x_unit = env.word_to_idx(x_unit, unit_input=True)

            ### Greedy solution.
            (
                generations,                                                                                                #(slen, bs)
                generations_len, 
                generations_units,                                                                                          #(slen, bs, 5) if has
                word_perplexity,                                                                                            #(bs,)
                units_perplexity                                                                                            #(bs,) if has
            ) = decoder.generate(
                src_enc = encoded,
                src_len = x_len,
                decode_physical_units = env.params.decode_physical_units,
                max_len=self.max_generated_output_len,
                sample_temperature=None,
                units = x_unit
            )

            #reshape
            slen = generations.shape[0]
            generations = generations.unsqueeze(-1).view(slen, bs, 1)                                                       #(slen, bs, 1)
            generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()                                        #(bs, 1, slen) (list)
            if generations_units is not None:
                generations_units = generations_units.unsqueeze(-1).view(slen, bs, 1, 5)                                    #(slen, bs, 1, 5)
                generations_units = generations_units.transpose(0, 1).transpose(1, 2).reshape((bs, 1, -1)).cpu().tolist()   #(bs, 1, slen*5) (list)
            else:
                generations_units = [[None] for _ in range(bs)]                                                             #(bs, 1, ) (list)
            word_perplexity = word_perplexity.unsqueeze(-1).view(bs, 1).cpu().tolist()                                      #(bs, 1, ) (list)
            if units_perplexity is not None:
                units_perplexity = units_perplexity.unsqueeze(-1).view(bs, 1).cpu().tolist()                                #(bs, 1, ) (list)
            else:
                units_perplexity = [[None] for _ in range(bs)]                                                              #(bs, 1, ) (list)

            
            #idx to expr
            generations = [
                list(
                    env.equation_encoder.decode(
                        lst=[env.equation_id2word[int(term)] for term in hyp1[1:-1]], 
                        xy_units=[units[idx] for idx in chunk][i],
                        decode_physical_units=env.params.decode_physical_units,
                        units=[env.equation_id2word[int(term)] for term in hyp2[5:-5]] if hyp2 is not None else None,
                        lable_unit_fn=env.generator.label_units,
                    )
                    for hyp1, hyp2 in zip(generations[i], generations_units[i])
                )
                for i in range(bs)
            ]

            #filter
            for i in range(bs):
                idx_to_remove = [generations[i][j] is None for j in range(1)]

                generations[i] =            [x for j,x in enumerate(generations[i])         if not idx_to_remove[j]]
                word_perplexity[i] =        [x for j,x in enumerate(word_perplexity[i])     if not idx_to_remove[j]]
                units_perplexity[i] =       [x for j,x in enumerate(units_perplexity[i])    if not idx_to_remove[j]]


            if self.beam_type == "search":
                raise NotImplementedError()
                _, _, search_generations = decoder.generate_beam(
                    encoded,
                    x_len,
                    beam_size=self.beam_size,
                    length_penalty=self.beam_length_penalty,
                    max_len=self.max_generated_output_len,
                    early_stopping=self.beam_early_stopping,
                )
                search_generations = [
                    sorted(
                        [hyp for hyp in search_generations[i].hyp],
                        key=lambda s: s[0],
                        reverse=True,
                    )
                    for i in range(bs)
                ]
                search_generations = [
                    list(
                        filter(
                            lambda x: x is not None,
                            [
                                env.idx_to_infix(
                                    hyp.cpu().tolist()[1:],
                                    is_float=False,
                                    str_array=False,
                                )
                                for (_, hyp) in search_generations[i]
                            ],
                        )
                    )
                    for i in range(bs)
                ]
                for i in range(bs):
                    generations[i].extend(search_generations[i])

            elif self.beam_type == "sampling":
                num_samples = self.beam_size
                encoded = (
                    encoded.unsqueeze(1)
                    .expand((bs, num_samples) + encoded.shape[1:])
                    .contiguous()
                    .view((bs * num_samples,) + encoded.shape[1:])
                )
                x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)

                #!!!TODO!!!
                sampling_x_unit = [u for u in unit.copy() for _ in range(num_samples)]           #BE CAREFUL!!!

                #sampling
                (
                    sampling_generations,                                                                                                #(slen, bs*num_samples)
                    sampling_generations_len, 
                    sampling_generations_units,                                                                                          #(slen, bs*num_samples, 5) if has
                    sampling_word_perplexity,                                                                                            #(bs*num_samples)
                    sampling_units_perplexity                                                                                            #(bs*num_samples) if has
                ) = decoder.generate(
                    src_enc = encoded,
                    src_len = x_len,
                    decode_physical_units = env.params.decode_physical_units,
                    max_len=self.max_generated_output_len,
                    sample_temperature=None,
                    units = sampling_x_unit
                )

                #reshape
                slen = sampling_generations.shape[0]
                sampling_generations = sampling_generations.unsqueeze(-1).view(slen, bs, num_samples)                                    #(slen, bs, num_samples)
                sampling_generations = sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()                               #(bs, num_samples, slen) (list)
                if sampling_generations_units is not None:
                    sampling_generations_units = sampling_generations_units.unsqueeze(-2).view(slen, bs, num_samples, 5)                 #(slen, bs, num_samples, 5)
                    sampling_generations_units = (
                        sampling_generations_units.transpose(0, 1).transpose(1, 2).reshape((bs, 1, -1)).cpu().tolist()
                    )                                                                                                                    #(bs, num_samples, slen*5) (list)
                else:
                    sampling_generations_units = [[None for _ in range(num_samples)] for _ in range(bs)]                                 #(bs, num_samples, ) (list)
                sampling_word_perplexity = sampling_word_perplexity.view(bs, num_samples).cpu().tolist()                                 #(bs, num_samples, ) (list)
                if sampling_units_perplexity is not None:
                    sampling_units_perplexity = sampling_units_perplexity.view(bs, num_samples).cpu().tolist()                           #(bs, num_samples, ) (list)
                else:
                    sampling_units_perplexity = [[None for _ in range(num_samples)] for _ in range(bs)]                                  #(bs, num_samples, ) (list)

                #idx to expr
                sampling_generations = [
                    list(
                        env.equation_encoder.decode(
                            lst=[env.equation_id2word[int(term)] for term in hyp1[1:-1]], 
                            xy_units=[units[idx] for idx in chunk][i],
                            decode_physical_units=env.params.decode_physical_units,
                            units=[env.equation_id2word[int(term)] for term in hyp2[5:-5]] if hyp2 is not None else None,
                            lable_unit_fn=env.generator.label_units,
                        )
                        for hyp1, hyp2 in zip(sampling_generations[i], sampling_generations_units[i])
                    )
                    for i in range(bs)
                ]

                #filter
                for i in range(bs):
                    idx_to_remove = [sampling_generations[i][j] is None for j in range(num_samples)]

                    sampling_generations[i] =       [x for j,x in enumerate(sampling_generations[i])          if not idx_to_remove[j]]
                    sampling_word_perplexity[i] =   [x for j,x in enumerate(sampling_word_perplexity[i])      if not idx_to_remove[j]]
                    sampling_units_perplexity[i] =  [x for j,x in enumerate(sampling_units_perplexity[i])     if not idx_to_remove[j]]

                    generations[i].extend           (sampling_generations[i])
                    word_perplexity[i].extend       (sampling_word_perplexity[i])
                    units_perplexity[i].extend      (sampling_units_perplexity[i])

                outputs[0].extend(generations)
                outputs[1].extend(word_perplexity)
                outputs[2].extend(units_perplexity)

            else:
                raise NotImplementedError
            
        return outputs
