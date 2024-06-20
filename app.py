import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    prediction = str(pred)
    
    return prediction


title = "Breast cancer detection with Deep Transfer Learning(ResNet18)."
description = "<p style='text-align: center'><b>As a radiologist or oncologist, it is crucial to know what is wrong with a breast x-ray image.<b><br><b>Upload the breast X-ray image to know what is wrong with a patients breast with or without inplant<b><p>"
article="<p style='text-align: center'>Web app is built and managed by Addai Fosberg<b></p>"
examples = ['img1.jpeg', 'img2.jpeg']
enable_queue=True

#interpretation='default'

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,enable_queue=enable_queue).launch()