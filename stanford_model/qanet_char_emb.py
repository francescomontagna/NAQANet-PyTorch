class CharQANet():
    def __init__(self, device,
                 char_emb_size = 200
                 word_emb_size:int = 300,
                 d_model:int = 128,
                 c_max_len: int = 500,
                 q_max_len: int = 300,
                 p_dropout: float = 0.1,
                 num_heads : int = 8):

    