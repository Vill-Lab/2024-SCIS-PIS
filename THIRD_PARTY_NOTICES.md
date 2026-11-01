# Third-party notices

The existing project MIT license is retained in LICENSE. Upstream code and publication assets retain their own attribution and applicable terms.

## Variational U-Net

The VAE architecture, graph utilities, perceptual loss and pose normalization are adapted from the supplied PIS implementation, which builds on CompVis/VUNet:

- Source: https://github.com/CompVis/vunet
- Paper: Patrick Esser, Ekaterina Sutter and Bjorn Ommer, A Variational U-Net for Conditional Appearance and Shape Generation, CVPR 2018.
- Project: https://compvis.github.io/vunet/
- A separate license file was not present in the supplied VUNet source or the inspected upstream root. This notice does not assign a new license to upstream material.

## PixelCNN++

The supplied nn.py identifies its TensorFlow operations as modified from PixelCNN++ (OpenAI). Original source: https://github.com/openai/pixel-cnn . The upstream MIT and Apache 2.0 license text is reproduced in licenses/PixelCNN-LICENSE.md.

## Attribute annotations

The matching utility reads user-supplied attributes and optionally the official Market1501 attribute MAT structure from https://github.com/vana77/Market-1501_Attribute . No images, identity annotations, pairing results or attribute-recognition weights are distributed in this repository.

## Paper framework figure

figures/pis-framework.jpg is the original Figure 2 of Person identity shift for privacy-preserving person re-identification, Science China Information Sciences 69, 112103 (2026), DOI https://doi.org/10.1007/s11432-023-4583-x . It was extracted as the original JPEG stream from page 6 of the official PDF, without redrawing, cropping or recompression. Publication attribution is retained; the project software license does not relicense this figure.

Official PDF: https://www.sciengine.com/sci-open/api/v1/open/file/pdfView/ECD81130A2924B9E9CAF7744D4330D2C

Figure SHA256: 7daf90f544777209d4d2aba495b4f4119a23650f1b0a4947f1c5865eeadfb359
