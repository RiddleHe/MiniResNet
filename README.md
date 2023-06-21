# MiniResNet

A deep residual neural network in the fashion of ResNet50, capable of completing image classification tasks efficiently. The core of the model, the identity blocks and the convolution blocks that have residual connections, is manually implemented.

## Technologies

- Implemented blocks of layers that have residual connections to solve the classic vanishing gradient problem during back propagation in a very deep model.

- Created convolution blocks that adjust the dimensions of the input with a convolution layer to support dimension-changing transformations within the block.

- Improved the performance of the model by creating 15 blocks of 10-layered residual connections to pick up high-level features in the input images.

## Performance

- Trained for 10 epochs on a A100 GPU with 25k images, decreasing the loss to around 0.5.
