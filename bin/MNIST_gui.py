import gradio as gr
import mnist as MNIST


def richiesta(image: str, compile_: bool):
    """
    The function that will be called when the button is pressed.

    Args:
        image (str): The path to the image
        compile_ (bool): Compile the model

    Returns:
        str: The result of the inference
    """
    MNIST.set_ic_gradio(False)
    device = MNIST.get_device()
    if image is not None:
        try:
            image = MNIST.preprocess(image).to(device)
        except Exception as e:
            print(f"Error preprocessin the image: {e}")

        model = MNIST.Net()
        model = MNIST.load_model(model, compile_)
        result = MNIST.infer(model, device, image)
        result = MNIST.postprocess(result)

    return str(result)


with gr.Blocks() as demo:
    gr.Markdown("# GUI visualization of the NN")
    gr.Markdown("### Drop the image you want to scan in the area below.")
    gr.Markdown("### The Image has to be black on white")
    image = gr.Image(type="filepath")
    compile_ = gr.Checkbox(label="Compile the model")
    result = gr.Textbox(label="Prediction of the NN")

    btn = gr.Button("Run")
    btn.click(fn=richiesta,
              inputs=[image, compile_],
              outputs=result)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
