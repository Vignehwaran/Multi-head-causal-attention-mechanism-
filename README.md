# Multi-head-causal-attention-mechanism

𝐈 𝐜𝐫𝐞𝐚𝐭𝐞𝐝 𝐚 𝐜𝐨𝐦𝐩𝐥𝐞𝐭𝐞 𝐰𝐨𝐫𝐤𝐟𝐥𝐨𝐰 𝐨𝐟 𝐌𝐮𝐥𝐭𝐢-𝐇𝐞𝐚𝐝 𝐂𝐚𝐮𝐬𝐚𝐥 𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧, 𝐭𝐡𝐞 𝐜𝐨𝐫𝐞 𝐦𝐞𝐜𝐡𝐚𝐧𝐢𝐬𝐦 𝐮𝐬𝐞𝐝 𝐢𝐧 𝐆𝐏𝐓-𝟐 𝐟𝐨𝐫 𝐩𝐫𝐨𝐜𝐞𝐬𝐬𝐢𝐧𝐠 𝐬𝐞𝐪𝐮𝐞𝐧𝐜𝐞𝐬

High-level steps

1.Inputs: (1, 3, 6)

2.Weight matrices for projections: W_q, W_k, W_v (shapes based on embed_dim) — e.g., (6, 6) each

3.linear projections → produce Q, K, V: (1, 3, 6)

Reshape to group by tokens & heads: (1, 3, num_heads, head_dim) → e.g., (1, 3, 2, 3)

Transpose to group by heads: (1, num_heads, seq_len, head_dim) → e.g., (1, 2, 3, 3)

4.Compute attention scores: scores = Q @ Kᵀ (transpose last two dims)

5.Apply causal mask (prevent attending to future tokens) based on context size

6.Scale and normalize (softmax) the attention scores

7.Multiply attention weights by V; then reverse the reshape/transpose to reconstruct token grouping: (1, 3, num_heads, head_dim)

8.Concatenate heads → (1, 3, embed_dim) (e.g., (1, 3, 6))

9.Optional final linear projection to ensure input/output dimension match

10.Output: (1, 3, 6)
