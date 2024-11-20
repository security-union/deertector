use anyhow::Result;
use tch::{
	Device,
	Kind,
	nn::VarStore,
	vision::{
		imagenet,
		resnet::resnet18,
	}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Specify the device (CPU or CUDA)
    let mut vs = VarStore::new(Device::cuda_if_available());
	let model = resnet18(&vs.root(), 1000);
	vs.load("/Users/darioalessandro/Documents/deertector/models/model.safetensors")?;

   // Load the image file and resize it to the usual imagenet dimension of 224x224.
	let image = imagenet::load_image_and_resize224("/Users/darioalessandro/Documents/deertector/models/dogo.jpeg")?
    .to_device(vs.device());

    // Apply the forward pass of the model to get the logits
    let output = image
        .unsqueeze(0)
        .apply_t(&model, false)
        .softmax(-1, Kind::Float);

    // Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }

    Ok(())
}
