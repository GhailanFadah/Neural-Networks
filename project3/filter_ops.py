'''filter_ops.py
Implements the convolution and max pooling operations.
Applied to images and other data represented as an ndarray.
YOUR NAMES HERE
CS343: Neural Networks
Project 3: Convolutional neural networks
'''
import numpy as np
import math


def conv2_gray(img, kers, verbose=True):
    '''Does a 2D convolution operation on GRAYSCALE `img` using kernels `kers`.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    img: ndarray. Grayscale input image to be filtered. shape=(height img_y (px), width img_x (px))
    kers: ndarray. Convolution kernels. shape=(Num kers, ker_sz (px), ker_sz (px))
        NOTE: Kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all the `kers`. shape=(Num kers, img_y, img_x)

    Hints:
    -----------
    - Remember to flip your kernel since this is convolution!
    - Be careful of off-by-one errors, especially in setting up your loops. In particular, you
    want to align your convolution so that it starts aligned with the top-left corner of your
    padded image and iterate until the right/bottom sides of the kernel fall in the last pixel on
    the right/bottom sides of the padded image.
    - Use the 'same' padding formula for compute the necessary amount of padding to have the output
    image have the same spatial dimensions as the input.
    - I suggest using indexing/assignment to 'frame' your input image into the padded one.
    '''
    img_y, img_x = img.shape
    n_kers, ker_x, ker_y = kers.shape
    
    if verbose:
        print(f'img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_x={ker_x}, ker_y={ker_y}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    f_Img = np.zeros((n_kers, img_y, img_x))
    
    padding = math.ceil((ker_x - 1) /2)

    rows = np.zeros((img_y, padding))
    
    a = np.hstack((rows, img))
    
    b = np.hstack((a, rows)) 
    
    cols= np.zeros((padding, b.shape[1]))
    
    c = np.vstack((b, cols))
    
    padded_image = np.vstack((cols, c)) 
    
    
    for k in range(n_kers):
        kernel = kers[k]
        kernel = np.flipud(np.fliplr(kernel))
  
    
        for i in range(img_y):
            for j in range(img_x):
                region = padded_image[i:i + ker_x, j:j + ker_x]
            
                result = np.sum(region * kernel)
                
                f_Img[k, i, j] = result
            
    return f_Img

def conv2(img, kers, verbose=True):
    '''Does a 2D convolution operation on COLOR or grayscale `img` using kernels `kers`.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    img: ndarray. Input image to be filtered. shape=(N_CHANS, height img_y (px), width img_x (px))
        where n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(Num filters, ker_sz (px), ker_sz (px))
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all `kers`. shape=(Num filters, N_CHANS, img_y, img_x)

    What's new:
    -----------
    - N_CHANS, see above.

    Hints:
    -----------
    - You should not need more for loops than you have in `conv2_gray`.
    - When summing inside your nested loops, keep in mind the keepdims=True parameter of np.sum and
    be aware of which axes you are summing over. If you use keepdims=True, you may want to remove
    singleton dimensions.
    '''
    n_chan, img_y, img_x = img.shape
    n_kers, ker_x, ker_y = kers.shape
    
    if verbose:
        print(f'img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_x={ker_x}, ker_y={ker_y}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    f_Img = np.zeros((n_kers, n_chan, img_y, img_x))
    
    padding = math.ceil((ker_x - 1) /2)

    padded_img = np.pad(img,( (0,0), (padding, padding), (padding, padding)))

    if verbose:
        print("f_Img shape; ",f_Img.shape)
        print("img shape:",img.shape)
        print("padding: ",padding)
        print("padded_image shape:",padded_img.shape)

    for k in range(n_kers):
        #we do this for each kernel
        kernel = kers[k]
        kernel = np.flipud(np.fliplr(kernel))
        for i in range(img_y):
            for j in range(img_x):
                #step into region
                region = padded_img[:,i:i + ker_y, j:j + ker_x]
                #do I need to sum with some axis.  It seems like our region pulls us into one of our dimensions
                #so why would me need to do anything but sum across the element wise operated matrix
                result = np.sum(region * kernel, axis = (1,2))
                f_Img[k,:, i, j] += result
            
    return f_Img
    



def conv2nn(imgs, kers, bias, verbose=True):
    '''General 2D convolution operation suitable for a convolutional layer of a neural network.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    imgs: ndarray. Input IMAGES to be filtered. shape=(BATCH_SZ, n_chans, img_y, img_x)
        where batch_sz is the number of images in the mini-batch
        n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(n_kers, N_CHANS, ker_sz, ker_sz)
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    bias: ndarray. Bias term used in the neural network layer. Shape=(n_kers,)
        i.e. there is a single bias per filter. Convolution by the c-th filter gets the c-th
        bias term added to it.
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    What's new (vs conv2):
    -----------
    - Multiple images (mini-batch support)
    - Kernels now have a color channel dimension too
    - Collapse (sum) over color channels when computing the returned output images
    - A bias term

    Returns:
    -----------
    output: ndarray. `imgs` filtered with all `kers`. shape=(BATCH_SZ, n_kers, img_y, img_x)

    Hints:
    -----------
    - You may need additional loop(s).
    - Summing inside your loop can be made simpler compared to conv2.
    - Adding the bias should be easy.
    '''
    batch_sz, n_chans, img_y, img_x = imgs.shape
    n_kers, n_ker_chans, ker_x, ker_y = kers.shape

    if verbose:
        print(f'batch_sz={batch_sz}, n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, n_ker_chans={n_ker_chans}, ker_x={ker_x}, ker_y={ker_y}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return

    if n_chans != n_ker_chans:
        print('Number of kernel channels doesnt match input num channels!')
        return
    
    f_Imgs = np.zeros((batch_sz, n_kers, img_y, img_x))
    
    padding = math.ceil((ker_x - 1) /2)

    padded_imgs = np.pad(imgs,((0,0),(0,0), (padding, padding), (padding, padding)))

    if verbose:
        print("f_Img shape; ",f_Imgs.shape)
        print("img shape:",imgs.shape)
        print("padding: ",padding)
        print("padded_image shape:",padded_imgs.shape)

    flipped_kers = np.flip(kers, axis=(2, 3))

    for img in range(batch_sz):
        for k in range(n_kers):
            #we do this for each kernel
            # kernel = kers[k]
            # kernel = np.flip(kernel, axis=(1, 2))#np.flipud(np.fliplr(kernel))
            for i in range(img_y):
                for j in range(img_x):
                    #step into region
                    region = padded_imgs[img,:,i:i + ker_y, j:j + ker_x]
                    result = np.sum(region * flipped_kers[k]) #will be n_channels
                    f_Imgs[img,k, i, j] += result + bias[k]
                
    return f_Imgs


def get_pooling_out_shape(img_dim, pool_size, strides):
    '''Computes the size of the output of a max pooling operation along one spatial dimension.

    Parameters:
    -----------
    img_dim: int. Either img_y or img_x
    pool_size: int. Size of pooling window in one dimension: either x or y (assumed the same).
    strides: int. Size of stride when the max pooling window moves from one position to another.

    Returns:
    -----------
    int. The size in pixels of the output of the image after max pooling is applied, in the dimension
        img_dim.
    '''

    return int(np.floor((img_dim - pool_size) / strides) + 1)


def max_pool(inputs, pool_size=2, strides=1, verbose=True):
    ''' Does max pooling on inputs. Works on single grayscale images, so somewhat comparable to
    `conv2_gray`.

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(height img_y, width img_x)
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that? :)

    NOTE: There is no padding in the max-pooling operation.

    Hints:
    -----------
    - You should be able to heavily leverage the structure of your conv2_gray code here
    - Instead of defining a kernel, indexing strategically may be helpful
    - You may need to keep track of and update indices for both the input and output images
    - Overall, this should be a simpler implementation than `conv2_gray`
    '''
    
    img_y, img_x = inputs.shape

    out_dims = get_pooling_out_shape(img_y, pool_size, strides),get_pooling_out_shape(img_x, pool_size, strides)

    output = np.zeros((out_dims[0], out_dims[1]))

    for i in range(out_dims[0]):
        for j in range(out_dims[1]):
            region = inputs[strides*i:strides*i + pool_size, strides*j:strides*j + pool_size]
            output[i,j] = np.max(region)

    
    return output


def max_poolnn(inputs, pool_size=2, strides=1, verbose=True):
    ''' Max pooling implementation for a MaxPooling2D layer of a neural network

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(mini_batch_sz, n_chans, height img_y, width img_x)
        where n_chans is 1 for grayscale images and 3 for RGB color images
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(mini_batch_sz, n_chans, out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that?)

    What's new (vs max_pool):
    -----------
    - Multiple images (mini-batch support)
    - Images now have a color channel dimension too

    Hints:
    -----------
    - If you added additional nested loops, be careful when you reset your input image indices
    '''
    mini_batch_sz, n_chans, img_y, img_x = inputs.shape

    #we can do this vectorized

    out_dims =mini_batch_sz, n_chans, get_pooling_out_shape(img_y, pool_size, strides),get_pooling_out_shape(img_x, pool_size, strides)

    output = np.zeros((mini_batch_sz, n_chans, out_dims[2], out_dims[3]))

    for i in range(out_dims[2]):
        for j in range(out_dims[3]):
            region = inputs[:,:,strides*i:strides*i + pool_size, strides*j:strides*j + pool_size]
            output[:,:,i,j] = np.max(region, axis = (2,3))

    
    return output


def conv2nn_transpose(input, wts, b, verbose=False):
    """
    Transposed convolution operation.

    Parameters:
    -----------
    input: ndarray. Input data. shape=(batch_sz, n_chans, img_y, img_x)
    wts: ndarray. Filter weights. shape=(n_kers, n_ker_chans, ker_sz, ker_sz)
    b: ndarray. Bias. shape=(n_kers,)
    verbose: bool. Whether to print debug info.

    Returns:
    -----------
    net_in: ndarray. Output of the transposed convolution. shape=(batch_sz, n_kers, out_y, out_x)
    """
    batch_sz, n_chans, img_y, img_x = input.shape
    n_kers, n_ker_chans, ker_sz, _ = wts.shape

    if verbose:
        print("conv2nn_transpose:")
        print(f'batch_sz={batch_sz}, n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, n_ker_chans={n_ker_chans}, ker_sz={ker_sz}')

    # output size
    out_y = img_y * 2
    out_x = img_x * 2

    # initialize the output with zeros
    net_in = np.zeros((batch_sz, n_kers, out_y, out_x))

    for n in range(batch_sz):
        for k in range(n_kers):
            for y in range(img_y):
                for x in range(img_x):
                        # step into region and apply weights
                        net_in[n, :, y:y+ker_sz, x:x+ker_sz] += input[n, k, y, x] * wts[k]


    # Add bias
    for k in range(n_kers):
        net_in[:, k, :, :] += b[k]

    return net_in
