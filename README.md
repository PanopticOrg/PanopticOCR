# PanopticOCR

A simple plugin to perform OCR on the selected images, this will create a new property OCRPlugin containing the ocrized text.
This plugin can be configured to use different models:
- default model is using [docTR](https://github.com/mindee/doctr) to perform the OCR, it can work on complex documents and also perform HTR but is quite slow
- for faster OCR you can also got to the params of the plugin in panoptic and set the model to "tesseract" make sure to install tesseract manually before: https://tesseract-ocr.github.io/tessdoc/Installation.html note: OCR is usually not as good as doctr but performs correctly on simple documents.

You can also change the params to crop the images if needed.

Warning: it can take a lot of time to perform OCR on a lot of images. 

