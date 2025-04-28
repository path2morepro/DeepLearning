import tensorflow as tf
import tf_keras
from tf_keras import layers, Model
from tf_keras.layers import (
    Add,
    Dense,
    Dropout,
    Input,
    Layer,
    LayerNormalization,
    Embedding,
)

import matplotlib.pyplot as plt
import numpy as np


# seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)


def scaled_dot_product_attention(q, k, v):
    """
    Function to compute the scaled dot-product attention.
    In this implementation we are accounting also for the number of heads,
    which will be useful when implementing the MultiHeadAttention layer.

    This will not change our implementation much; what you need to take into consideration
    is that the attention is computed over the last two dimensions of the input vectors.

    Parameters:
        q (tf.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, depth)
        k (tf.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, depth)
        v (tf.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, depth)

    Returns:
        output (tf.Tensor): output tensor of shape (batch_size, num_heads, seq_len_q, depth)
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------

    # Compute the dot product between queries and keys.
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 第一个问题： 这个维度怎么感觉匹配不上啊？
    # 仅仅在最后两个维度做transpose
    # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

    # Scale the dot products by the square root of the depth.
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # 第二个问题： 我好像取depth取不出来啊
    # 好多操作都是tensorflow的内置函数......
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the softmax function to obtain the attention weights (use tf.nn.softmax).
    attention_weights = tf.nn.softmax(scaled_attention_logits)  
    # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

    # Compute the weighted sum of the values.
    output = tf.matmul(attention_weights, v)
    # 第三个问题：这里怎么不transpose?
    # attention_weights: (bs, heads, seq_len_q, sen_len_k)
    # v: (batch_size, num_heads, seq_len_v, depth)
    # seq_len_v == seq_len_k 所以刚好
    # Shape: (batch_size, num_heads, seq_len_q, depth)

    # ============================================

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, n_heads, proj_size, dk, dv, return_attention_weights=False):
        super(MultiHeadAttention, self).__init__()
        """
        Builds the MultiHeadAttention layer. 

        Parameters:
            n_heads (int): number of attention heads
            proj_size (int): dimension of the projected vectors
            dk (int): dimension of the keys
            dv (int): dimension of the values
            return_attention_weights (bool): whether to return the attention weights
        """

        super(MultiHeadAttention, self).__init__()

        # define initializer function
        initializer = tf.keras.initializers.GlorotUniform()
        # 这是干嘛的？
        # 参数初始化类

        # --------------------------------------------
        # === Your code here =========================
        # --------------------------------------------

        # Initialize the weights for the projection layers. Note that here we are combining all the heads together.
        # Remember to set the trainable parameter to True
        self.WQ = tf.Variable(initializer(shape=(n_heads, dk, proj_size)), trainable=True)  # of shape (n_heads, proj_size, dq=dk)
        self.WK = tf.Variable(initializer(shape=(n_heads, dk, proj_size)), trainable=True)  # of shape (n_heads, proj_size, dk=dq)
        self.WV = tf.Variable(initializer(shape=(n_heads, dv, proj_size)), trainable=True)  # of shape (n_heads, proj_size, dv)
        self.WO = tf.Variable(initializer(shape=(n_heads*proj_size, proj_size)), trainable=True) # of shape (n_heads*proj_size, proj_size)
        # project_size到底代表什么啊？
        # project_size：输入向量的原始维度
        # dv/dk：投影后的每头的小维度
        # 注释写错了妈的
 
        # ============================================

        # print(f"WQ shape: {self.WQ.shape}")
        # print(f"WK shape: {self.WK.shape}")
        # print(f"WV shape: {self.WV.shape}")
        # print(f"WO shape: {self.WO.shape}")

        self.return_attention_weights = return_attention_weights

    def call(self, Q, K, V):
        """
        Forward pass of the MultiHeadAttention layer

        Parameters:
            Q (tf.Tensor): Query tensor of shape (batch_size, number_of_Q, dq=dk)
            K (tf.Tensor): Key tensor of shape (batch_size, number_of_K, dk=dq)
            V (tf.Tensor): Value tensor of shape (batch_size, number_of_V, dv)

        Returns:
            output (tf.Tensor): Multihead attention pooling of shape (batch_size, number_of_Q, projection_dimension)
            A (tf.Tensor): Attention weights of shape (batch_size, number_of_Q, number_of_K)

            NOTE : the output of the MultiHeadAttention is not dv but instead projection_dimension.
            This is to facilitate the skip connections that will be used in the full attention layer.
        """

        # --------------------------------------------
        # === Your code here =========================
        # --------------------------------------------

        # Projecting Q,K,V to Qh, Kh, Vh. The H projection are stacked on the along the second-to-last axis.
        # NOTE : here one needs to use tf.experimental.numpy.dot instead of tf.matmul as the former supports broadcasting.

        # Qh: (bs, num_of_Q, dq)
        # WQ: (n_heads, proj_size, dq)
        Qh = tf.experimental.numpy.dot(Q, self.WQ)  # of shape (batch_size, number_of_Q, n_heads, dk=dq)
        Kh = tf.experimental.numpy.dot(K, self.WK)  # of shape (batch_size, number_of_K, n_heads, dk=dq)
        Vh = tf.experimental.numpy.dot(V, self.WV)  # of shape (batch_size, number_of_V, n_heads, dv)
        
        # Bring the number of queries, keys, and their dimension to the last two axes so that we can use the scaled_dot_product_attention function
        # 这一步点积其实和n_heads没有关系，只是为了将dq消除而已，就是矩阵乘法中间的维度消失了,dp就应该是中间的维度
        Qh = tf.transpose(Qh, perm=[0, 2, 1, 3])  # of shape (batch_size, H, number_of_Q, proj_size)
        Kh = tf.transpose(Kh, perm=[0, 2, 1, 3])  # of shape (batch_size, H, number_of_K, proj_size)
        Vh = tf.transpose(Vh, perm=[0, 2, 1, 3])  # of shape (batch_size, H, number_of_V, proj_size)
        # print(f"Qh shape: {tf.shape(Qh)}")
        # print(f"Kh shape: {tf.shape(Kh)}")
        # print(f"Vh shape: {tf.shape(Vh)}")
        # Computing the dot-product attention
        attention_pooling_h, attention_weights_h = scaled_dot_product_attention(Qh, Kh, Vh)  # of shape (batch_size, n_heads, number_of_Q, proj_size)

        # Flattening (concatenate) across the number of heads.
        batch_size = tf.shape(attention_pooling_h)[0]
        n_heads =  tf.shape(attention_pooling_h)[1]
        num_of_Q =  tf.shape(attention_pooling_h)[2]
        project_size =  tf.shape(attention_pooling_h)[3]
        A = tf.reshape(attention_pooling_h, (batch_size, num_of_Q, n_heads * project_size))  # of shape (batch_size, number_of_Q, n_heads*proj_dim)

        # Projecting the concatenated heads to the output space
        A = tf.experimental.numpy.dot(A, self.WO)  # of shape (batch_size, number_of_Q, proj_dim)

        # ============================================

        if self.return_attention_weights:
            return A, attention_weights_h
        else:
            return A


def mlp(x, hidden_units, dropout_rate):
    """
    Function that passes the input thorough a multi-layer perceptron (MLP).
    Part of the TransformerBlock.

    Parameters:
        x (tf.Tensor): input tensor
        hidden_units (list): list of integers specifying the number of units in each dense layer
        dropout_rate (float): dropout rate

    Returns:
        x (tf.Tensor): output tensor
    """
    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Input:
    # (batch_size, number_of_Q, proj_dim)
    # (batch_size, ...,  2048)
    # (batch_size, number_of_Q,  proj_dim)
    # 问题：中间维度是怎样变化的？或者说，dense是如何使得输入数据维度产生改变的
    # 维度改变仅仅是因为W矩阵，那么我觉得中间维度还是num_of_Q

    # 问题： 之前定义dense都是先定义再编译再训练，这次直接要求的是输出，怎么办？
    # 定义好直接调用

    # 问题：两层的units不一样啊......
    # 如下...
    for i, units in enumerate(hidden_units):
        if i == len(hidden_units) - 1:
            x = Dense(units, activation=None)(x)
        else:
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout_rate)(x)

    # 问题：Transformer的MLP最后一层不是不用激活函数么？
    # 是的，所以改一下原来的结构吧
    # ============================================
    return x


def transformerBlock(x, num_heads, projection_dim, transformer_units, dropout_rate=0.1):
    """
    Function to create a transformer block

    Parameters:
        x (tf.Tensor): input tensor (these are the encoded patches in our implementation)
        num_heads (int): number of attention heads
        projection_dim (int): dimension of the projected vectors
        transformer_units (list): list of integers specifying the number of units in each dense layer of the MLP
        dropout_rate (float): dropout rate

    Returns:
        x2 (tf.Tensor): output tensor
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # 那么dv和dk怎么整呢？目前还不知道x的尺寸
    # dv,dk应该是x的最后一维
    dk = x.shape[-1]
    # apply the multi-head attention
    # x (batch_size, num_patches, embedding_dim)
    # print(f"shape of x is {tf.shape(x)}")
    # attention_output (batch_size, number_of_Q, proj_dim)
    # num_patches == number_of_Q
    attention_output = MultiHeadAttention(num_heads, projection_dim, dk, dk)(x,x,x) 
    # print(f"shape of attention_output is {tf.shape(attention_output)}")
    # NOTE: here we are using the MultiHeadAttention layer to compute the SELF ATTENTION
    # 他意思是这里直接返回计算后的结果？所以调用call()
    # note的意思其实是：SELF ATTENTION， 所以传递参数的话，x,x,x
    

    # apply dropout
    attention_output = Dropout(dropout_rate)(attention_output)
    # attention还有dropout?
    # 防止注意力过于集中
    
    # apply the skip connection and layer normalization
    x1 = attention_output + x
    x1 = LayerNormalization(axis=-1)(x1)
    # 这个batch normalization是不是加的不对
    # 问题：对哪一个维度进行layerNormalization？
    # 对最后一个维度进行标准化
    # 问题：skip是从哪里到哪里的skip?
    
    # apply the MLP
    # x1 (batch_size, number_of_Q, proj_dim)
    # 经过中间层的x (batch_size, ...,  2048)
    # mlp_output (batch_size, number_of_Q,  proj_dim)
    # mlp parameters (x1, hidden_units, dropout_rate)
    mlp_output = mlp(x1, transformer_units, dropout_rate)
    # print(f"shape of mlp output is {tf.shape(mlp_output)}")
    # 这里调用怎么不用加类名
    #　问题：dense 的权重矩阵的维度和num_of_hidden有什么联系？
    #　权重矩阵 W 的维度是 (input_dim, num_of_hidden)

    # apply dropout
    mlp_output = Dropout(dropout_rate)(mlp_output)

    # apply the skip connection and layer normalization
    # 我觉得这个加的也不对
    x2 = mlp_output + x1
    x2 = LayerNormalization(axis=-1)(x2)

    # 这里和图片中的transformer结构不同
    # 采用post-layernorlization: 在skpi connection之后添加layernormalization
    # 图片中的结构采用pre-layernormalization
    # 有什么区别呢？
    # 后者适用于深层模型，避免梯度问题
    # ============================================
    return x2


class PatchExtractor(Layer):
    def __init__(self, patch_size=16):
        """
        Initializes the PatchExtractor layer.

        Parameters:
            patch_size (int): size of the patches to be extracted
        """
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        """
        Forward pass of the PatchExtractor layer.

        Parameters:
            images (tf.Tensor): input tensor of shape (batch_size, height, width, channels)

        Returns:
            patches (tf.Tensor): output tensor of shape (batch_size, num_patches, patch_dims)
        """

        # --------------------------------------------
        # === Your code here =========================
        # --------------------------------------------
        # Get batch size
        batch_size = tf.shape(images)[0]

        # Use the tf.image.extract_patches function to extract patches from the images (see documentation for details: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches)
        size = [1, self.patch_size, self.patch_size, 1]
        patches = tf.image.extract_patches(images=images,
                                            sizes=size, 
                                            strides=size,
                                            rates=[1, 1, 1, 1],
                                            padding='VALID')
        # strides == sizes 不重叠不跳过
        # 问题：返回值维度是多少？
        # patches都被展平（flatten）到最后一个通道维度里去了
        # patches: (batch_size, 2, 2, 16*16*3)
        # 最后一维就是一个patch的embedding

        # get the dimensions of the patches tensor
        # 问题：这是在求什么？和patch_size的区别是？
        # patch最后被embedding成了一个一维向量，现在在求这个向量的长度
        patch_dims = patches.shape[-1]
        num_patches = patches.shape[1] * patches.shape[2]
        # reshape the patches tensor to have the correct shape (batch_size, num_patches, patch_dims)
        patches = tf.reshape(tensor=patches, shape=[batch_size, num_patches, patch_dims])
        # 设置成-1不会因为input层自动初始化batch size = None而报错
        # ============================================
        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches: int = 4, projection_dim: int = 768):
        """
        Initializes the PatchEncoder layer.

        Parameters:
            num_patches (int): number of patches
            projection_dim (int): dimension of the projected vectors
        """
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        # initialize the class token with a random normal initializer
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        # 这个为什么是trainable?

        # initialize the projection layer and the position embedding layer
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches + 1, output_dim=projection_dim
        )

    def call(self, patch):
        # --------------------------------------------
        # === Your code here =========================
        # --------------------------------------------

        # get the batch size
        batch_size = tf.shape(patch)[0]
        # print(f"batch_size shape {batch_size}")
        # 这里又要使用tf.shape取维度
        # 因为batch_size是动态的
        # 在测试里面等于None

        # Reshape the class token embeddings (first make as many copies as the batch size and then reshape)
        # CLS: (1, projection_dim) -> CLS:(batch_size, 1, projection_dim)
        # Use the tf.tile function to make as many copies of the class token as the batch size (see documentation for details: https://www.tensorflow.org/api_docs/python/tf/tile)
        # print(f"class token size before expand {tf.shape(self.class_token)}")
        class_token = tf.tile(self.class_token, [batch_size, 1])
        # print(f"class token size after expand {class_token.shape}")
        class_token = tf.reshape(tensor=class_token, shape=[batch_size, 1, self.projection_dim])  # shape: (batch_size, 1, projection_dim)
        # print(f"class token size after reshape {class_token.shape}")
        # Project the patches using the projection layer
        # 投影 patch embedding
        # 把每个 patch 向量从原始的维度映射到 projection_dim
        patches_embed = self.projection(patch)
        # patches_embed = patch

        # concatenate the class token to the patches
        patches_embed = tf.concat(values=[patches_embed, class_token], axis=1)

        # calculate positional embeddings based on the number of patches (how many positions?)
        # 问题：位置编码是在给谁的位置编码，是patch在image中的位置么？
        # 是的
        num_of_positions = patch.shape[1] + 1
        positions = tf.range(num_of_positions)
        # position: (num_of_positions)
        positions_embed = self.position_embedding(positions)
        # positions_embed: (num_of_positions, projection_dim)

        # 问题：这里直接相加么？
        # 先扩展 positions_embed 使其形状变成 [1, num_of_positions, projection_dim]
        positions_embed = tf.expand_dims(positions_embed, axis=0)
        # 然后用tile扩展到 [batch_size, num_of_positions, projection_dim]
        positions_embed = tf.tile(positions_embed, [batch_size, 1, 1])
        # add the positional embeddings to the patch embeddings
        # print(f"positions_embed {tf.shape(positions_embed)}")
        # print(f"patches_embed {tf.shape(patches_embed)}")
        encoded = patches_embed + positions_embed

        # ============================================

        return encoded


def create_vit_classifier(
    input_shape: tuple = (32, 32, 3),
    patch_size: int = 16,
    data_augmentation=None,
    embedding_proj_dim: int = 768,
    num_heads: int = 12,
    msa_proj_dim: int = 768,
    transformer_layers: int = 12,
    msa_dropout_rate: float = 0.1,
    mlp_classification_head_units: list = [3072],
    mlp_classification_head_dropout_rate: float = 0.5,
    num_classes: int = 10,
):
    """
    Function to create a Vision Transformer (ViT) classifier model.

    Parameters:
        input_shape (tuple): input shape of the images
        patch_size (int): size of the patches to be extracted
        data_augmentation (tf.keras.Sequential): data augmentation pipeline
        embedding_proj_dim (int): dimension of the projected vectors
        num_heads (int): number of attention heads
        msa_proj_dim (int): dimension of the projected vectors in the MultiHeadAttention layer
        transformer_layers (int): number of transformer layers
        msa_dropout_rate (float): dropout rate in the MultiHeadAttention layer
        mlp_classification_head_units (list): list of integers specifying the number of units in each dense layer of the MLP
        mlp_classification_head_dropout_rate (float): dropout rate in the classification head MLP
        num_classes (int): number of classes

    Returns:
        model (tf.keras.Model): Vision Transformer model

    """
    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------

    # Define input layer
    inputs = Input(input_shape)
    # print(f"input shape is {inputs.shape}")
    # Augment data (if provided)
    if data_augmentation is not None:
        inputs = data_augmentation(inputs)

    # Create patches.
    patches_extractor = PatchExtractor(patch_size=patch_size)
    # print(f"input shape is {inputs.shape}")
    patches = patches_extractor(inputs)

    # Encode patches.
    ## Calculate the number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    #这样取维度会引发错误
    # tf.shape返回值是运行时的动态形状
    # 只能现场计算了

    ## Encode the patches using the PatchEncoder layer
    patch_encoder = PatchEncoder(num_patches=num_patches, projection_dim=embedding_proj_dim)
    encoded_patches = patch_encoder(patches)

    # Create multiple layers of the Transformer block
    ## define mlp transformer units based on the msa_proj_dim
    transformer_units = [
        msa_proj_dim * 2,
        msa_proj_dim,
    ]  # Size of the transformer layers

    for _ in range(transformer_layers):
        # 问题: 在实现问题中,多层transformer是怎么实现的
        # 或者具体一点, 他在循环什么?
        encoded_patches = transformerBlock(x=encoded_patches,
                                            num_heads=num_heads, 
                                            projection_dim=msa_proj_dim,
                                            transformer_units=transformer_units,
                                            dropout_rate=msa_dropout_rate)

    # Take out the class token (it is the last token)
    representation = encoded_patches[:, -1, :]

    # classification head applied to the class token
    ## Add mpl
    # 问题: 在transformerblock中定义过mlp之后最后出来还是要经过mlp分类?
    # 是的
    features = mlp(x=representation,
                   hidden_units=mlp_classification_head_units,
                   dropout_rate=mlp_classification_head_dropout_rate)

    ## features to the number of classes
    logits = Dense(num_classes)(features)

    # ============================================

    # Create the Keras model.
    model = Model(inputs=inputs, outputs=logits)
    return model


# #################### TEST FUNCTIONS ####################
# Define a test function for the scaled_dot_product_attention where for a given set of fixed 10 by 5 matrices Q, K, and V, the function returns the attention weights and the output of the scaled dot-product attention.
def test_scaled_dot_product_attention(function=scaled_dot_product_attention, seed=42):
    # seed everything
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # test name
    test_name = "scaled_dot_product_attention"

    # Define Q, K, and V matrices to be used for testing.
    # Define the input tensors
    batch_size = 2
    nub_q = 10
    nub_k = 5

    dk = 5
    dv = 5

    Q = tf.random.normal((batch_size, nub_q, dk))
    K = tf.random.normal((batch_size, nub_k, dk))
    V = tf.random.normal((batch_size, nub_k, dv))

    print("############################################")
    print(f"### Testing {test_name} ###")
    print("############################################")
    print(f"Random seed set to: {seed} (expecting 42)")

    if seed != 42:
        print(
            "ATTENTION! Random seed not set correctly (test designed for random seed == 42)."
        )

    print(f"Testing on Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

    # define stats of the expected output and attention weights
    expected_mean_output = 0.1913447231054306
    expected_std_output = 0.5758092403411865

    expected_mean_attention_weights = 0.20000000298023224
    expected_std_attention_weights = 0.15885186195373535

    # Compare the output and attention weights to the expected values
    output, attention_scores = function(Q, K, V)

    output_mean_difference = tf.reduce_mean(output).numpy() - expected_mean_output
    output_std_difference = tf.math.reduce_std(output).numpy() - expected_std_output

    attention_weights_mean_difference = (
        tf.reduce_mean(attention_scores).numpy() - expected_mean_attention_weights
    )
    attention_weights_std_difference = (
        tf.math.reduce_std(attention_scores).numpy() - expected_std_attention_weights
    )

    # TODO : check that the random seed works on several machines.
    if (
        output_mean_difference == 0.0
        and output_std_difference == 0.0
        and attention_weights_mean_difference == 0.0
        and attention_weights_std_difference == 0.0
    ):
        print(f"### {test_name} test passed ###")
        print("Difference with expected values == 0.0")
    else:
        print(f"### {test_name} test failed ###")

        print(f"Output mean difference: {output_mean_difference} (expected  0.0).")
        print(f"Output std difference: {output_std_difference} (expected  0.0).")
        print(
            f"Attention weights mean difference: {attention_weights_mean_difference} (expected  0.0)."
        )
        print(
            f"Attention weights std difference: {attention_weights_std_difference} (expected  0.0)."
        )
    print("############################################\n\n")


# define a test function for the MultiHeadAttention layer
def test_MultiHeadAttention(seed=42):
    # seed everything
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # test name
    test_name = "MultiHeadAttention"

    # Define the MultiHeadAttention layer
    n_heads = 2
    proj_size = 8
    dk = 5
    dv = 5
    return_attention_weights = True
    multi_head_attention = MultiHeadAttention(
        n_heads, proj_size, dk, dv, return_attention_weights
    )

    # Define the input tensors
    batch_size = 2
    nub_q = 10
    nub_k = 5

    dk = 5
    dv = 5

    Q = tf.random.normal((batch_size, nub_q, dk))
    K = tf.random.normal((batch_size, nub_k, dk))
    V = tf.random.normal((batch_size, nub_k, dv))

    print("###############################################")
    print(f"### Testing {test_name} definition ###")
    print("###############################################")
    print(f"Random seed set to: {seed} (expecting 42)")
    if seed != 42:
        print(
            "ATTENTION! Random seed not set correctly (test designed for random seed == 42)."
        )

    print(f"Testing on Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(
        f"Number of heads: {n_heads}, projection size: {proj_size}, dk: {dk}, dv: {dv}"
    )

    # Define the expected outputs
    expected_mean_output = -0.019008774310350418
    expected_std_output = 0.4834766983985901

    expected_mean_attention_weights = 0.20000000298023224
    expected_std_attention_weights = 0.08575260639190674

    # Get output and attention weights from the MultiHeadAttention layer
    output, attention_scores = multi_head_attention(Q, K, V)

    # Compare the output and attention weights to the expected values
    output_mean_difference = tf.reduce_mean(output).numpy() - expected_mean_output
    output_std_difference = tf.math.reduce_std(output).numpy() - expected_std_output

    attention_weights_mean_difference = (
        tf.reduce_mean(attention_scores).numpy() - expected_mean_attention_weights
    )
    attention_weights_std_difference = (
        tf.math.reduce_std(attention_scores).numpy() - expected_std_attention_weights
    )

    # TODO : check that the random seed works on several machines.
    if (
        output_mean_difference == 0.0
        and output_std_difference == 0.0
        and attention_weights_mean_difference == 0.0
        and attention_weights_std_difference == 0.0
    ):
        print(f"\n### {test_name} test passed ###")
        print("Difference with expected values == 0.0")
    else:
        print(f"### {test_name} test failed ###")

        print(f"Output mean difference: {output_mean_difference} (expected  0.0).")
        print(f"Output std difference: {output_std_difference} (expected  0.0).")
        print(
            f"Attention weights mean difference: {attention_weights_mean_difference} (expected  0.0)."
        )
        print(
            f"Attention weights std difference: {attention_weights_std_difference} (expected  0.0)."
        )
        # print expected shapes for the intermediate tensors in the MultiHeadAttention layer for this configuration
        print("\n")
        print(f"Expected shape of WQ: (2, 5, 8)")
        print(f"Expected shape of WK: (2, 5, 8)")
        print(f"Expected shape of WV: (2, 5, 8)")
        print(f"Expected shape of WO: (16, 8)")
        print("\n")
        print(f"Expected shape of Qh: (2, 2, 10, 8)")
        print(f"Expected shape of Kh: (2, 2, 5, 8)")
        print(f"Expected shape of Vh: (2, 2, 5, 8)")
        print("\n")
        print(f"Expected shape of attention_pooling_h: (2, 2, 10, 8)")
        print(f"Expected shape of A: (2, 10, 8)")
        print(f"Expected shape of attention_weights: (2, 10, 5)")

    print("###############################################")


# define a test function for the TransformerBlock
def test_TransformerBlock(seed=42, dropout_rate=0.1):
    # seed everything
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # test name
    test_name = "TransformerBlock"

    # specify the TransformerBlock layer inputs
    n_heads = 2
    proj_size = 32

    transformer_units = [
        proj_size * 2,
        proj_size,
    ]

    # Define the input tensors
    batch_size = 2
    num_x = 10
    dx = proj_size

    X = tf.random.normal((batch_size, num_x, dx))

    print("###########################################")
    print("### Testing transformerBlock definition ###")
    print("###########################################")
    print(
        f"Random seed set to: {seed} (expecting 42), dropout rate: {dropout_rate} (expected 0.1)"
    )
    if seed != 42:
        print(
            "ATTENTION! Random seed not set correctly (test designed for random seed == 42)."
        )
    if dropout_rate != 0.1:
        print(
            "ATTENTION! Dropout rate not set correctly (test designed for dropout rate == 0.1)."
        )

    print(f"Testing on Q: {X.shape}")
    print(
        f"Number of heads: {n_heads}, projection size: {proj_size}, dx: {dx}, transformer_units: {transformer_units}"
    )

    # Define the expected outputs
    expected_mean_output = 0.0
    expected_std_output = 1

    # Get output and attention weights from the MultiHeadAttention layer
    output = transformerBlock(X, n_heads, proj_size, transformer_units)

    # Compare the output and attention weights to the expected values
    output_mean_difference = tf.reduce_mean(output).numpy() - expected_mean_output
    output_std_difference = tf.math.reduce_std(output).numpy() - expected_std_output

    # TODO : check that the random seed works on several machines.
    if output_mean_difference < 0.0001 and output_std_difference < 0.0001:
        print(f"\n### {test_name} test passed ###")
        print("Difference with expected values == 0.0")
    else:
        print(f"### {test_name} test failed ###")

        print(f"Output mean difference: {output_mean_difference} (expected  0.0).")
        print(f"Output std difference: {output_std_difference} (expected  0.0).")
    print("###############################################")


# #################### PLOTTING FUNCTIONS ####################


def plot_training_examples(images, labels, class_names):
    """
    Function to plot a grid of images with their corresponding labels

    Parameters:
        images (np.array): images of shape (height, width, 3)
        labels (np.array): labels of shape (num_samples,)
        class_names (list): list of class names

    Returns:
        None
    """
    plt.figure(figsize=(12, 4))

    for i in range(18):
        idx = np.random.randint(7500)
        label = labels[idx, 0]

        plt.subplot(3, 6, i + 1)
        plt.tight_layout()
        plt.imshow(images[idx])
        plt.title("Class: {} ({})".format(label, class_names[label]))
        plt.axis("off")
    plt.show()


def plot_original_and_patched_version(original_image, patches, patch_size):
    """

    Parameters:
        original_image (tf.Tensor): original image tensor of shape (height, width, 3)
        patches (tf.Tensor): patches of the original image obtained through the PatchExtractor layer of shape (batch_size, num_patches, patch_dims)
    Returns:
        None
    """

    # get the number of rows and columns in the patch subplot
    nbr_patches_sqrt = int(np.sqrt(patches.shape[1]))

    # create figure and subplots
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(1, 2).flat

    # Original image
    subfig = subfigs[0]
    ax = subfig.subplots(1, 1)
    # convert image [-1, 1] to [0, 255] integer
    original_image = ((original_image + 1) * 127.5).astype(np.uint8)
    ax.imshow(original_image)
    ax.set_title("Original Image")
    ax.axis("off")

    # Patched image
    subfig = subfigs[1]
    subfig.suptitle(f"Patches")
    axs = subfig.subplots(nbr_patches_sqrt, nbr_patches_sqrt)
    for i, ax in enumerate(axs.flat):
        # reshape to patch size by patch size by 3
        image_patch = patches[0, i].numpy().reshape(patch_size, patch_size, 3)
        # convert image [-1, 1] to [0, 255] integer
        image_patch = ((image_patch + 1) * 127.5).astype(np.uint8)
        ax.imshow(image_patch)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_classifier_training_history(history):
    """
    Function to plot the training history of a model

    Parameters:
        history (tf.keras.History): history object of a trained model

    Returns:
        None
    """

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    item = "loss"
    plt.plot(history.history[item], label="Training loss")
    plt.plot(history.history["val_" + item], label="Validation loss")
    plt.grid()
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # check if the history has a key for accuracy
    if "accuracy" in history.history:
        plt.subplot(1, 2, 2)
        item = "accuracy"
        plt.plot(history.history[item], label="Training Accuracy")
        plt.plot(history.history["val_" + item], label="Validation Accuracy")
        plt.grid()
        plt.title("Training and Validation Accuracy", fontsize=16)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    plt.show()
