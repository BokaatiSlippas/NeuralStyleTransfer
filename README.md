# NeuralStyleTransfer
Doing neural style transfer crap


https://www.youtube.com/watch?v=6KGtaXR7yMU&ab_channel=NeilRhodes

https://www.youtube.com/watch?v=c4vuR4vHKd0&ab_channel=NeilRhodes

https://www.youtube.com/watch?v=AJOyMJjPDtE&ab_channel=NeilRhodes

Notes:

calculate activations of I_content and I_comb on the final (or maybe a bit before) layer of the pretrained VGG-19 conv NN and do mse on these activations as the Loss(I_content)

dot product of maps gives relative importance stylistically for the cooccurrence of different maps

calculate gram matrices for each selected layer for styles
Loss_style(I_comb) = average Loss_style(I_comb, a[l]) for each selected layer (FUTURE NOTE: maybe different levels of significance per layer)
this is the same as 


https://arxiv.org/pdf/1508.06576
