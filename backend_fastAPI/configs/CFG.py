"""
The project and model config in json format
"""

CFG = {
    "augmentation": {
        "path":"img_for_test/Augmentation/img.JPG",
        "show": True,
        "rot_angle": 30,
        "illum_level": 1.5,
        "zoom": 5,
        "cont_factor":1.5
    },
	"data": {
		"path": "data/",
		"image_size": 256,
		"load_with_info": True,
		"shuffle":True,
		"seed":42,
		"val_split":0.2,
		"batch_size": 32,
	},
	"train": {
		"batch_size": 32,
		"epoches": 15,
		"optimizer": "adam",
		"metrics": ['accuracy']
	},
	"model": {
		"input_shape": [256, 256, 3],
		"output_channels": 8,
		"save_name": "saved_models/my_cnn_model.keras"
	},
    "predict": {
        "path":"img_for_test/Augmenation/img.JPG",
        "show_result_plot":True
    }
}