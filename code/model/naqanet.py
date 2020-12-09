import math
import tqdm
import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from code.modules.encoder.encoder import EncoderBlock
from code.modules.encoder.depthwise_conv import DepthwiseSeparableConv
from code.modules.pointer import Pointer
from code.modules.cq_attention import CQAttention
from code.modules.embeddings import Embedding
from code.modules.utils import set_mask
from code.drop_eval.drop_metric import eval_dicts, convert_tokens
from code.util import (torch_from_json, masked_softmax, get_best_span, \
 replace_masked_values_with_big_negative_number)
from code.model.qanet import QANet
from code.args_drop import get_train_args
from code.dataset.drop import collate_fn, DROP
import code.util as util
from code import util

# Debug only
EVAL_EXAMPLE =  dict()
EVAL_EXAMPLE['0'] = {'context': " Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.", 'question': 'Who scored the first touchdown of the game?',
 'spans': [(0, 1), (1, 7), (8, 10), (11, 18), (19, 23), (24, 29), (30, 34), (35, 37), (38, 41), (42, 50), (50, 51), (52, 55), (56, 63), (64, 70), (71, 73), (74, 78), (79, 82), (83, 84), (85, 89), (90, 92), (93, 97), (98, 102), (103, 106), (107, 114), (115, 121), (121, 122), (122, 123), (124, 131), (132, 137), (138, 141), (142, 145), (146, 151), (152, 156), (157, 159), (160, 163), (164, 169), (170, 177), (178, 180), (181, 192), (193, 201), (202, 209), (210, 219), (220, 221), (222, 229), (230, 239), (240, 244), (245, 247), (248, 254), (255, 259), (260, 268), (269, 273), (274, 282), (282, 283), (283, 284), (285, 288), (289, 295), (296, 301), (302, 309), (310, 314), (315, 323), (324, 329), (330, 335), (336, 343), (344, 345), (346, 352), (353, 362), (363, 366), (366, 367), (368, 371), (372, 375), (376, 383), (384, 389), (390, 396), (397, 401), (402, 408), (409, 418), (419, 429), (430, 437), (438, 439), (440, 447), (448, 451), (452, 453), (454, 461), (462, 467), (468, 472), (472, 473), (473, 474), (475, 482), (483, 488), (489, 492), (493, 496), (497, 501), (502, 504), (505, 508), (509, 515), (516, 523), (524, 528), (529, 535), (536, 540), (541, 546), (547, 554), (555, 556), (557, 564), (565, 568), (569, 570), (571, 578), (579, 584), (585, 589), (589, 590), (591, 598), (599, 604), (605, 609), (610, 613), (614, 618), (619, 621), (622, 625), (626, 631), (632, 639), (640, 644), (645, 649), (650, 658), (659, 666), (667, 670), (671, 678), (679, 687), (688, 689), (690, 697), (698, 707), (708, 712), (713, 717), (718, 725), (725, 726), (727, 735), (736, 738), (739, 741), (742, 744), (745, 752), (753, 757), (758, 764), (765, 768), (769, 770), (771, 780), (780, 781), (781, 782), (783, 786), (787, 793), (794, 799), (800, 802), (803, 808), (809, 811), (812, 815), (816, 822), (823, 830), (831, 833), (834, 839), (840, 846), (847, 848), (849, 856), (857, 862), (863, 867), (867, 868), (869, 872), (873, 876), (877, 884), (884, 885), (886, 893), (894, 899), (900, 904), (905, 909), (910, 913), (914, 922), (923, 930), (930, 931)],
 'answer': {'number': '', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': ['Chaz Schilens']}}
EVAL_EXAMPLE['1'] = {'context': "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007", 'question': 'How many points did the buccaneers need to tie in the first?',
 'spans': [(0, 2), (3, 8), (9, 12), (13, 19), (19, 20), (21, 24), (25, 30), (31, 39), (40, 45), (46, 48), (49, 54), (54, 55), (56, 63), (64, 66), (67, 71), (72, 74), (75, 78), (79, 84), (85, 88), (89, 99), (99, 100), (101, 104), (105, 110), (111, 117), (118, 123), (124, 126), (127, 130), (131, 136), (137, 144), (145, 149), (150, 151), (152, 159), (160, 165), (166, 170), (171, 173), (174, 179), (180, 186), (186, 187), (188, 191), (192, 202), (203, 207), (208, 210), (211, 213), (214, 218), (219, 220), (221, 228), (229, 234), (235, 239), (240, 242), (243, 249), (250, 255), (255, 256), (257, 261), (262, 266), (267, 270), (271, 275), (276, 280), (281, 285), (286, 291), (292, 303), (304, 305), (306, 310), (311, 315), (316, 323), (324, 332), (333, 336), (337, 340), (341, 343), (344, 346), (347, 349), (350, 355), (355, 356), (357, 360), (361, 366), (367, 376), (377, 381), (382, 383), (384, 391), (392, 397), (398, 402), (402, 403), (404, 406), (407, 410), (411, 417), (418, 425), (425, 426), (427, 434), (435, 439), (440, 443), (444, 448), (449, 453), (454, 455), (456, 463), (464, 473), (474, 479), (480, 482), (483, 489), (490, 497), (497, 498), (499, 502), (503, 508), (509, 514), (515, 519), (520, 526), (527, 531), (532, 536), (537, 546), (547, 553), (554, 556), (557, 564), (565, 567), (568, 572), (572, 573), (574, 579), (580, 583), (584, 593), (594, 598), (599, 600), (601, 608), (609, 614), (615, 619), (620, 624), (625, 631), (632, 640), (640, 641), (642, 645), (646, 652), (653, 657), (658, 661), (662, 672), (673, 678), (678, 679), (680, 684), (685, 689), (690, 694), (695, 699), (700, 707), (708, 711), (712, 721), (721, 722), (723, 728), (728, 729), (730, 737), (737, 739), (740, 746), (747, 754), (755, 761), (762, 763), (764, 770), (771, 775), (776, 778), (779, 782), (783, 788), (789, 796), (796, 797), (798, 801), (802, 806), (806, 808), (809, 814), (815, 821), (822, 826), (827, 831), (832, 836), (837, 845), (846, 848), (849, 854), (855, 858), (859, 865), (866, 867), (868, 874), (875, 879), (879, 880), (880, 881), (882, 885), (886, 891), (892, 895), (896, 901), (902, 909), (910, 916), (917, 923), (924, 927), (928, 931), (932, 937), (938, 942), (943, 948), (949, 953)],
 'answer': {'number': '0', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}}


class NAQANet(QANet):
    def __init__(self, 
                 device,
                 word_embeddings,
                 char_embeddings,
                 w_emb_size:int = 300,
                 c_emb_size:int = 64,
                 hidden_size:int = 128,
                 c_max_len: int = 800,
                 q_max_len: int = 100,
                 p_dropout: float = 0.1,
                 num_heads : int = 8, 
                 answering_abilities = ['passage_span_extraction', 'counting', 'addition_subtraction'],
                 max_count = 10): # max number the network can count
        """
        :param hidden_size: hidden size of representation vectors
        :param q_max_len: max number of words in a question sentence
        :param c_max_len: max number of words in a context sentence
        :param p_dropout: dropout probability
        """
        super().__init__(
            device, 
            word_embeddings,
            char_embeddings,
            w_emb_size,
            c_emb_size,
            hidden_size,
            c_max_len,
            q_max_len,
            p_dropout,
            num_heads)

        # Implementing numerically augmented output for QANet
        self.answering_abilities = answering_abilities
        self.max_count = max_count

        # Initialize eval_data to None
        self.eval_data = None

        # passage and question representations coefficients
        self.passage_weights_layer = nn.Linear(hidden_size, 1)
        self.question_weights_layer = nn.Linear(hidden_size, 1)

        # TODO fix
        self.modeling_encoder_blocks = nn.ModuleList([EncoderBlock(device, hidden_size, len_sentence=c_max_len, p_dropout=0.1) \
                                             for _ in range(2)])

        # answer type predictor
        if len(self.answering_abilities) > 1:
            self.answer_ability_predictor = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                # nn.ReLU(), 
                nn.Dropout(p = self.p_dropout),
                nn.Linear(hidden_size, len(self.answering_abilities)),
                # nn.ReLU(), 
                nn.Dropout(p = self.p_dropout)
            ) # then, apply a softmax
        

        if 'passage_span_extraction' in self.answering_abilities:
            self.passage_span_extraction_index = self.answering_abilities.index(
                "passage_span_extraction"
            )
            self.passage_span_start_predictor = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                # nn.ReLU(), 
                nn.Linear(hidden_size, 1),
                # nn.ReLU()
            )
            self.passage_span_end_predictor = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                # nn.ReLU(), 
                nn.Linear(hidden_size, 1),
                # nn.ReLU() 
            ) # then, apply a softmax

        if 'counting' in self.answering_abilities:
            self.counting_index = self.answering_abilities.index("counting")
            self.count_number_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(), 
                nn.Dropout(p = self.p_dropout),
                nn.Linear(hidden_size, self.max_count),
                # nn.ReLU()
            ) # then, apply a softmax
        
        if 'addition_subtraction' in self.answering_abilities:
            self.addition_subtraction_index = self.answering_abilities.index(
                "addition_subtraction"
            )
            self.number_sign_predictor = nn.Sequential(
                nn.Linear(hidden_size*3, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, 3),
                # nn.ReLU()
            )

    def set_eval_data(self, gold_dict):
        self.eval_data = gold_dict

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs, ids,
                answer_start_as_passage_spans: torch.LongTensor = None,
                answer_end_as_passage_spans: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                number_indices = None):

        batch_size = cw_idxs.size(0)

        spans_start, spans_end = super().forward(cw_idxs, cc_idxs, qw_idxs, qc_idxs)

        # The first modeling layer is used to calculate the vector representation of passage
        passage_weights = masked_softmax(self.passage_weights_layer(self.passage_aware_rep).squeeze(-1), self.c_mask_c2q, log_softmax = False)
        passage_vector_rep = passage_weights.unsqueeze(1).bmm(self.passage_aware_rep).squeeze(1)
        # The second modeling layer is use to calculate the vector representation of question
        question_weights = masked_softmax(self.question_weights_layer(self.qb).squeeze(-1), self.q_mask_c2q, log_softmax = False)
        question_vector_rep = question_weights.unsqueeze(1).bmm(self.qb).squeeze(1)

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = self.answer_ability_predictor(
                torch.cat([passage_vector_rep, question_vector_rep], -1)
            )
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            # Shape: (batch_size,)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        if "counting" in self.answering_abilities:
            # Shape: (batch_size, self.max_count)
            count_number_logits = self.count_number_predictor(passage_vector_rep)
            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1) # softmax over possible numbers
            # Info about the best count number prediction
            # Shape: (batch_size,)
            best_count_number = torch.argmax(count_number_log_probs, -1) # most probable numeric value
            best_count_log_prob = torch.gather(
                count_number_log_probs, 1, best_count_number.unsqueeze(-1)
            ).squeeze(-1)
            
            if len(self.answering_abilities) > 1:
                best_count_log_prob += answer_ability_log_probs[:, self.counting_index]

        # Sto buttando il mio tempo
        if "addition_subtraction" in self.answering_abilities:
            # M3
            modeled_passage = self.modeled_passage_list[-1]
            for block in self.modeling_encoder_blocks:
                modeled_passage = self.dropout_layer(
                    block(modeled_passage, self.c_mask_enc)
                )
            
            self.modeled_passage_list.append(modeled_passage)
            encoded_passage_for_numbers = torch.cat(
                [self.modeled_passage_list[0], self.modeled_passage_list[3]], dim=-1
            )

            # Reshape number indices to (batch_size, # numbers in longest passage)
            
            # create mask on indices
            number_mask = number_indices != -1
            # print(f"number_mask {number_mask}")
            clamped_number_indices = number_indices.masked_fill(~number_mask, 0).type(torch.int64).to(self.device)
            number_mask = number_mask.to(self.device)

            if number_mask.size(1) > 0:
                # Shape: (batch_size, max_len_context, 3*hidden_size)
                encoded_numbers = torch.cat(
                    [
                        encoded_passage_for_numbers,
                        passage_vector_rep.unsqueeze(1).repeat(1, encoded_passage_for_numbers.size(1), 1),
                    ],
                    -1,
                )

                # Shape: (batch_size, max # number in passages, 3*hidden_size)
                encoded_numbers = torch.gather(encoded_numbers,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(
                        -1, -1, encoded_numbers.size(-1)
                    ))

                number_sign_logits = self.number_sign_predictor(encoded_numbers)
                number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)
                # print(f"number_sign_log_probs 1: {number_sign_log_probs}")

                # Shape: (batch_size, # of numbers in passage).
                best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
                # For padding numbers, the best sign masked as 0 (not included).
                best_signs_for_numbers = best_signs_for_numbers.masked_fill(~number_mask, 0)

                # TODO fix or remove
                # print(f"best_signs_for_numbers 2: {best_signs_for_numbers}") # Per qualche motivo se True mi restituisce sempre 2


                # Shape: (batch_size, # of numbers in passage)
                best_signs_log_probs = torch.gather(
                    number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1)
                ).squeeze(-1)

                # the probs of the masked positions should be 1 so that it will not affect the joint probability
                # TODO: this is not quite right, since if there are many numbers in the passage,
                # TODO: the joint probability would be very small.
                best_signs_log_probs = best_signs_log_probs.masked_fill(~number_mask, 0)
                # print(f"best_signs_log_probs 3: {best_signs_log_probs}")

                # Shape: (batch_size,)
                best_combination_log_prob = best_signs_log_probs.sum(-1)
                
                if len(self.answering_abilities) > 1:
                    best_combination_log_prob += answer_ability_log_probs[
                        :, self.addition_subtraction_index
                    ]

                else:
                    print("No numbers in the batch")

        
        # Both paper and code of naqanet implementation differ from paper and code of qanet...
        if "passage_span_extraction" in self.answering_abilities:
            # Shape: (batch_size, passage_length, modeling_dim * 2))
            passage_for_span_start = torch.cat(
                [self.modeled_passage_list[0], self.modeled_passage_list[1]], dim=-1
            )
            # Shape: (batch_size, passage_length)
            passage_span_start_logits = self.passage_span_start_predictor(
                passage_for_span_start
            ).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)
            passage_for_span_end = torch.cat(
                [self.modeled_passage_list[0], self.modeled_passage_list[2]], dim=-1
            )
            # Shape: (batch_size, passage_length)
            passage_span_end_logits = self.passage_span_end_predictor(
                passage_for_span_end
            ).squeeze(-1)
            # Shape: (batch_size, passage_length). Prob on log scale from -infinite to 0
            passage_span_start_log_probs = util.masked_log_softmax(
                passage_span_start_logits, self.c_mask_c2q
            )
            passage_span_end_log_probs = util.masked_log_softmax(
                passage_span_end_logits, self.c_mask_c2q
            )

            # Info about the best passage span prediction
            passage_span_start_logits = replace_masked_values_with_big_negative_number( \
                passage_span_start_logits, self.c_mask_c2q
            )
            passage_span_end_logits = replace_masked_values_with_big_negative_number(
                passage_span_end_logits, self.c_mask_c2q
            )
            # Shape: (batch_size, 2)
            best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
                
            # Shape: (batch_size, 2)
            best_passage_start_log_probs = torch.gather(
                passage_span_start_log_probs, 1, best_passage_span[:, 0].unsqueeze(-1)
            ).squeeze(-1)
            best_passage_end_log_probs = torch.gather(
                passage_span_end_log_probs, 1, best_passage_span[:, 1].unsqueeze(-1)
            ).squeeze(-1)
            # Shape: (batch_size,)
            best_passage_span_log_prob = best_passage_start_log_probs + best_passage_end_log_probs
            if len(self.answering_abilities) > 1:
                best_passage_span_log_prob += answer_ability_log_probs[
                    :, self.passage_span_extraction_index
                ]

        output_dict = dict()

        # If answer is given, compute the loss.
        if (
            answer_start_as_passage_spans is not None
            or answer_as_add_sub_expressions is not None
            or answer_as_counts is not None
        ):

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_start_as_passage_spans
                    gold_passage_span_ends = answer_end_as_passage_spans
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = gold_passage_span_starts != -1 # start and end should share same mask
                    clamped_gold_passage_span_starts = gold_passage_span_starts. \
                            masked_fill(~gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = gold_passage_span_ends. \
                            masked_fill(~gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = torch.gather(
                        passage_span_start_log_probs, 1, clamped_gold_passage_span_starts
                    )
                    log_likelihood_for_passage_span_ends = torch.gather(
                        passage_span_end_log_probs, 1, clamped_gold_passage_span_ends
                    )
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = (
                        log_likelihood_for_passage_span_starts
                        + log_likelihood_for_passage_span_ends
                    )
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = (
                        replace_masked_values_with_big_negative_number(
                            log_likelihood_for_passage_spans,
                            gold_passage_span_mask,
                        )
                    )
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(
                        log_likelihood_for_passage_spans
                    )

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)
                
                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = answer_as_counts != -1
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = answer_as_counts.masked_fill(~gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(
                        count_number_log_probs, 1, clamped_gold_counts
                    )
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = replace_masked_values_with_big_negative_number(
                        log_likelihood_for_counts, gold_count_mask
                    )
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")
            
            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = (
                    all_log_marginal_likelihoods + answer_ability_log_probs
                )
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]

            output_dict["loss"] = -marginal_log_likelihood.mean()

        if self.eval_data:
            output_dict["predictions"] = dict()
            for i in range(batch_size):

                id = ids[i].item()
                if len(self.answering_abilities) > 1:
                        predicted_ability_str = self.answering_abilities[
                            best_answer_ability[i].detach().cpu().numpy()
                        ]
                        # print(f"Predicted ability: {predicted_ability_str}")
                else:
                    predicted_ability_str = self.answering_abilities[0]

                if predicted_ability_str == "passage_span_extraction":
                    start = best_passage_span[i, 0]
                    end = best_passage_span[i, 1]
                    preds = convert_tokens(self.eval_data,
                                           id,
                                           start.item(),
                                           end.item())
                    output_dict["predictions"][str(id)] = preds

                elif predicted_ability_str == "counting":
                    predicted_count = str(best_count_number[i].detach().cpu().numpy())
                    output_dict["predictions"][str(id)] = predicted_count


        return output_dict
        

if __name__ == "__main__":
    eval_debug = False
    train_debug = True
    debug_real_data = False # debug using train_dataloader
    torch.manual_seed(224)
    np.random.seed(224)

    if eval_debug or train_debug:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wemb_vocab_size = 5000
        number_emb_idxs = np.random.default_rng().choice(np.arange(1, wemb_vocab_size), size = int(wemb_vocab_size/4), replace = False)
        wemb_dim = 300
        cemb_vocab_size = 94
        cemb_dim = 64
        d_model = 128
        batch_size = 32
        q_max_len = 6
        c_max_len = 100
        spans_limit = 6
        num_limit = 5
        char_dim = 16
        max_count = 100000

        # fake embedding
        wv_tensor = torch.rand(wemb_vocab_size, wemb_dim)
        cv_tensor = torch.rand(cemb_vocab_size, cemb_dim)

        # fake input
        question_lengths = torch.LongTensor(batch_size).random_(1, q_max_len)
        question_wids = torch.zeros(batch_size, q_max_len).long()
        question_cids = torch.zeros(batch_size, q_max_len, char_dim).long()
        context_lengths = torch.LongTensor(batch_size).random_(1, c_max_len)
        context_wids = torch.zeros(batch_size, c_max_len).long()
        context_cids = torch.zeros(batch_size, c_max_len, char_dim).long()

        num_idxs_length = torch.LongTensor(batch_size).random_(0, num_limit)

        spans_length = torch.LongTensor(batch_size).random_(0, spans_limit)
        start_indices = torch.zeros(batch_size, spans_limit).long() -1
        end_indices = torch.zeros(batch_size, spans_limit).long() -1
        counts = torch.zeros(batch_size).long() -1

        ids = torch.tensor(range(0, batch_size))

        print("Example sizes")
        print(f"context_wids: {context_wids.size()}")
        print(f"context_cids: {context_cids.size()}")
        print(f"question_wids: {question_wids.size()}")
        print(f"question_cids: {question_cids.size()}")
        print(f"start_indices: {start_indices.size()}")
        print(f"end_indices: {end_indices.size()}")
        print(f"counts: {counts.size()}")
        print(f"ids: {ids.size()}")

        for i in range(batch_size):
            question_wids[i, 0:question_lengths[i]] = \
                torch.LongTensor(1, question_lengths[i]).random_(
                    1, wemb_vocab_size)
            question_cids[i, 0:question_lengths[i], :] = \
                torch.LongTensor(1, question_lengths[i], char_dim).random_(
                    1, cemb_vocab_size)
            context_wids[i, 0:context_lengths[i]] = \
                torch.LongTensor(1, context_lengths[i]).random_(
                    1, wemb_vocab_size)
            context_cids[i, 0:context_lengths[i], :] = \
                torch.LongTensor(1, context_lengths[i], char_dim).random_(
                    1, cemb_vocab_size)
            start_indices[i, 0:spans_length[i]] = \
                torch.LongTensor(1, spans_length[i]).random_(
                    0, c_max_len)
            end_indices[i, 0:spans_length[i]] = \
                torch.LongTensor(1, spans_length[i]).random_(
                    0, c_max_len)
            counts[i] = torch.LongTensor(1).random_(
                    -1, max_count)

        counts = counts.unsqueeze(-1)

        # Fake evaluation dictionary
        for i in range(2, batch_size):
            if i%2 == 0:
                EVAL_EXAMPLE[str(i)] = EVAL_EXAMPLE['0']
            else:
                EVAL_EXAMPLE[str(i)] = EVAL_EXAMPLE['1']


    if eval_debug:

        # define model
        device = 'cpu'
        model = NAQANet(device, wv_tensor, cv_tensor, 
            answering_abilities = ['passage_span_extraction', 'counting'], max_count=max_count)
        model.train()

        # train
        output_dict = model(context_wids, context_cids,
                       question_wids, question_cids, ids,
                       start_indices, end_indices, counts)

        loss = output_dict["loss"]
        print(f"Training Loss: {loss.item()}")


        # eval
        model.eval_data = EVAL_EXAMPLE
        output_dict = model(context_wids, context_cids,
                       question_wids, question_cids, ids,
                       start_indices, end_indices, counts)

        loss = output_dict["loss"]
        
        print(f"Output dictionary: {output_dict}\n")

        eval_dict = eval_dicts(model.eval_data, output_dict["predictions"])

        F1 = eval_dict["F1"]
        EM = eval_dict["EM"]

        print(f"F1: {F1}")
        print(F"EM: {EM}")



    if train_debug: # want loss close to 0

        args = get_train_args()

        # define model
        device = 'cpu'
        model = NAQANet(device, wv_tensor, cv_tensor, 
            answering_abilities = ['passage_span_extraction', 'counting'], max_count=max_count)
        model = model.to(device)
        model.train()
        ema = util.EMA(model, args.decay)

        # Get optimizer and scheduler
        lr = args.lr
        base_lr = 1.0
        warm_up = args.lr_warm_up_num
        params = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(lr=base_lr, betas=(args.beta1, args.beta2), eps=1e-7, weight_decay=3e-7, params=params)
        cr = lr / math.log(warm_up)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * math.log(ee + 1) if ee < warm_up else lr)

        # Train
        with torch.enable_grad():
            # tqdm(total=len(train_loader.dataset)) as progress_bar:
            for epoch in range(70):
                # Setup for forward
                context_wids = context_wids.to(device)
                context_cids = context_cids.to(device)
                question_wids = question_wids.to(device)
                question_cids = question_cids.to(device)
                start_indices = start_indices.to(device)
                end_indices = end_indices.to(device)
                counts = counts.to(device)
                ids = ids.to(device)
                optimizer.zero_grad()

                # Forward
                output_dict = model(context_wids, context_cids,
                    question_wids, question_cids, ids,
                    start_indices, end_indices, counts)

                loss = output_dict["loss"]
                loss_val = loss.item()
                print(f"Loss val {epoch}: {loss_val}")

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                ema(model, epoch)


    if debug_real_data:

        args = get_train_args()

        # define model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        word_vectors = util.torch_from_json(args.word_emb_file)
        char_vectors = util.torch_from_json(args.char_emb_file)
        model = NAQANet(device, word_vectors, char_vectors,
            c_max_len = args.context_limit,
            q_max_len = args.question_limit,
            answering_abilities = ['passage_span_extraction', 'counting'],
            max_count = args.max_count) # doesn't large max_count lead to meaningless probability?
        model = model.to(device)
        model.train()
        ema = util.EMA(model, args.decay)

        # Get optimizer and scheduler
        lr = args.lr
        base_lr = 1.0
        warm_up = args.lr_warm_up_num
        params = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(lr=base_lr, betas=(args.beta1, args.beta2), eps=1e-7, weight_decay=3e-7, params=params)
        cr = lr / math.log(warm_up)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * math.log(ee + 1) if ee < warm_up else lr)

        # Get data
        train_dataset = DROP(args.train_record_file)
        train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)

        epoch = 1

        # Train
        with torch.enable_grad():
            # tqdm(total=len(train_loader.dataset)) as progress_bar:
            for context_wids, context_cids, \
                    question_wids, question_cids, \
                    start_indices, end_indices, \
                    counts, ids in train_loader:

                # Setup for forward
                context_wids = context_wids.to(device)
                context_cids = context_cids.to(device)
                question_wids = question_wids.to(device)
                question_cids = question_cids.to(device)
                start_indices = start_indices.to(device)
                end_indices = end_indices.to(device)
                counts = counts.to(device)
                ids = ids.to(device)
                optimizer.zero_grad()

                # Forward
                output_dict = model(context_wids, context_cids,
                    question_wids, question_cids, ids,
                    start_indices, end_indices, counts)

                loss = output_dict["loss"]
                loss_val = loss.item()
                print(f"Loss val {epoch-1}: {loss_val}")

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                ema(model, epoch)

                epoch += 1