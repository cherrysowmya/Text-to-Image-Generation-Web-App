from distutils.log import debug
from auth_token import auth_token
from flask import Flask, render_template, request
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import pickle

Hugging_face = auth_token

app = Flask(__name__)

pipe = pickle.load(open("sd_pipeline.pkl","rb" ) )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def generate():
    prompt = request.form["input-text"]
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale = 8.5).images[0]
    
    image.save("testimage.jpg")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)