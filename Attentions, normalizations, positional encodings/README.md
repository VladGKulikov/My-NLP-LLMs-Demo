![image](https://github.com/VladGKulikov/LLMs-Large-Language-Models/assets/98630446/2052e034-49fc-4a2d-90cc-a67c481aeb6b)

From: https://t.me/AIexTime/68

A Brief Note on the Changes in Transformer Architecture Since 2017.
 
Reading articles about LLMs, one may come across phrases like 'we use the standard transformer architecture.' 
But what does 'standard' mean, and have there been any changes since the original paper was published? 
Let's look at the main significant architectural improvements for LLMs, using the language model (i.e., decoder-only) LLaMa-2 as an example:

— Post LayerNorm → Pre LayerNorm (https://arxiv.org/abs/2002.04745). 

This makes convergence more stable. Now, the process is such that the original embeddings simply pass through the decoder blocks, and "adjustments" from the FFN and Attention are added to them. 

— Sinusoidal Positional Encoding → RoPE (https://arxiv.org/abs/2104.09864). 

The method involves rotating the token embeddings by an angle that depends on the position. And it works well. Besides, the method has opened up a whole range of modifications for expanding the context to very large numbers.

— Activation Function ReLU → SwiGLU (https://arxiv.org/abs/2002.05202). 

Gated Linear Units (a family of methods to which SwiGLU belongs. It adds an operation of element-wise multiplication of matrices, one of which has passed through a sigmoid, thus controlling the intensity of the signal passing from the first matrix) slightly improve quality on a range of tasks.

— LayerNorm → RMSNorm (https://arxiv.org/abs/1910.07467). 

RMSNorm is computationally simpler but works with the same quality.

— Attention Modifications (https://arxiv.org/abs/2305.13245). 

For example, using one K-V pair of matrices at once for a group of Q matrices. This improvement mainly affects inference optimization. But there is also a huge number of methods aimed at reducing the quadratic complexity of the operation, and more about this here (https://t.me/AIexTime/18) and here (https://t.me/AIexTime/30).
