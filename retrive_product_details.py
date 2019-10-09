import os
import sys
import warnings
import numpy as np
from PIL import Image

warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


class RetrieveProductDetails:
    def __init__(self, input_image):
        self._image = input_image

    def get_product_details(self):
        path_to_frozen_graph = '../Product_Detection/product_model/frozen_inference_graph.pb'
        image = Image.open(self._image)
        image_np = np.array(image.getdata()).reshape((image.size[1], image.size[0], 3)).astype(np.uint8)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            with tf.Session() as sess:
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['detection_boxes', 'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0][0]
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)[0]

        product_width = output_dict['detection_boxes'][0] * image.size[1] + output_dict['detection_boxes'][2] * \
                        image.size[1]
        product_height = output_dict['detection_boxes'][1] * image.size[0] + output_dict['detection_boxes'][3] * \
                         image.size[0]
        product_dim = int(product_width + product_height)
        product_name = ''
        if output_dict['detection_classes'] == 1:
            product_name = 'colgate'
        elif output_dict['detection_classes'] == 2:
            product_name = 'marie_gold'
        elif output_dict['detection_classes'] == 3:
            product_name = 'bag'
        elif output_dict['detection_classes'] == 4:
            product_name = 'shoe'

        product_dict = {}

        if product_name == 'colgate':
            product_dict['Product Name'] = 'colgate'
            if product_dim in range(642, 842):
                product_dict['Weight/Size'] = 50
                product_dict['Price'] = 54
            elif product_dim in range(1482, 1682):
                product_dict['Weight/Size'] = 100
                product_dict['Price'] = 75
        elif product_name == 'marie_gold':
            product_dict['Product Name'] = 'marie_gold'
            if product_dim in range(382, 582):
                product_dict['Weight/Size'] = 50
                product_dict['Price'] = 26
            elif product_dim in range(1334, 1534):
                product_dict['Weight/Size'] = 100
                product_dict['Price'] = 50
        elif product_name == 'bag':
            product_dict['Product Name'] = 'bag'
            if product_dim in range(424, 624):
                product_dict['Weight/Size'] = 'Small'
                product_dict['Price'] = 560
            elif product_dim in range(1548, 1748):
                product_dict['Weight/Size'] = 'Medium'
                product_dict['Price'] = 840
            elif product_dim in range(2930, 3250):
                product_dict['Weight/Size'] = 'Big'
                product_dict['Price'] = 950
        elif product_name == 'shoe':
            product_dict['Product Name'] = 'shoe'
            if product_dim in range(1764, 1964):
                product_dict['Weight/Size'] = 'S'
                product_dict['Price'] = 600
            elif product_dim in range(3100, 3337):
                product_dict['Weight/Size'] = 'M'
                product_dict['Price'] = 800
            elif product_dim in range(5100, 5400):
                product_dict['Weight/Size'] = 'XL'
                product_dict['Price'] = 1200

        product_dict['Weight/Size'] = str(product_dict['Weight/Size'])
        return product_dict


if __name__ == '__main__':
    rpd = RetrieveProductDetails(sys.argv[1])
    rpd.get_product_details()
