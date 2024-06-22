import gradio as gr
import MNIST
import torch


def richiesta(image: str, compile: bool):
    MNIST.set_ic_gradio(False)
    device = MNIST.get_device()
    print(compile)
    if image != None:
        try:
            image = MNIST.preprocess(image).to(device)
        except Exception as e:
            print(f"Error preprocessin the image: {e}")

        model = MNIST.Net()
        model = MNIST.load_model(model, compile)
        result = MNIST.infer(model, device, image)
        result = MNIST.postprocess(result)
        
    return str(result)


with gr.Blocks() as demo:
    gr.Markdown("GUI visualization of the NN")
    image = gr.Image(type="filepath")
    compile = gr.Checkbox(label="Compile the model")
    result = gr.Textbox(label="Prediction of the NN")
    
    btn = gr.Button("Run")
    btn.click(fn=richiesta,
              inputs=[image, compile],
              outputs=result)


if __name__ == "__main__":
    demo.launch()