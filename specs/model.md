# SPEC: MultiTaskCelebNet
                                                                                                                                   

### PURPOSE  

 A CNN that jointly predicts facial binary attributes and celebrity identity from a single forward pass.
                                                                                                                                   

### INPUTS  

 x : Tensor of shape (batch_size, 3, H, W), values in [0, 1]


### OUTPUTS  

1. **binary_logits:**  

   * Tensor of shape (batch_size, 8).  

   * Raw logits for 8 binary attributes:  
     - No_Beard
     - Young
     - Mouth_Slightly_Open
     - Smiling
     - Male
     - Wavy_Hair
     - Black_Hair
     - Wearing_Hat 
                
   * NO sigmoid applied — the loss function (BCEWithLogitsLoss) handles it.

2. **class_logits:**  

   * Tensor of shape (batch_size, 501)  
   
   * Raw logits for 501 identity classes (500 celebrities + 1 "other").  
   
   * NO softmax applied — CrossEntropyLoss handles it.  
                                                                                                                                   

### ARCHITECTURE  

**backbone:**   shared CNN layers (conv → activation → maxpool, repeated)
            accepts configurable filters, kernel sizes, pool sizes, activations  

**binary_head:** nn.Linear(backbone_out_dim, 8)  

**class_head:** nn.Linear(backbone_out_dim, 501)  
                                                                                                                                   

### CONSTRUCTOR PARAMETERS  

**img_shape**      : _tuple_       — image shape (Channel, Height, Width)  

**conv_filters**   : _list[int]_   — number of output filters per conv layer 

**kernel_sizes**   : _list[tuple]_ — kernel (H, W) per conv layer 

**max_pool_sizes** : _list[tuple]_ — pool (H, W) per layer; (1,1) = no pooling 

**act_fs**         : _list[fn]_    — activation function per conv layer 

               
### INVARIANTS  

len(conv_filters) == len(kernel_sizes) == len(max_pool_sizes) == len(act_fs)  

All conv layers use padding='same' → _spatial dims only change via pooling_  

backbone_out_dim = conv_filters[-1] * (in_height // prod(pool_H)) * (in_width // prod(pool_W))  

forward() always returns a tuple of exactly 2 tensors — never a single tensor  

                                                                                                                                   
### LABEL CONVENTION (matches CelebDataset column order)  
labels[:, :8]       → _binary attributes_  → fed to binary_head loss  

labels[:, 8].long() → _celebrity\_id_       → fed to class_head loss  