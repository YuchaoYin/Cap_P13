from styx_msgs.msg import TrafficLight
import rospy
import yaml
import os
import cv2
import numpy as np
import tensorflow as tf

# for debug
DEBUG = False

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.classes = {1: TrafficLight.RED,
                        2: TrafficLight.YELLOW,
                        3: TrafficLight.GREEN,
                        4: TrafficLight.UNKNOWN}

        self.session = None
        self.model_graph = None

        conf = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(conf)

        model = os.path.dirname(os.path.realpath(__file__)) + self.config['model']
        self.load_model(model)

        self.img_counter = 0

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        traffic_light_id, prob = self.predict(image)

        if traffic_light_id is not None:
            rospy.logdebug("Class: %d, Probability: %f", traffic_light_id, prob)

        return traffic_light_id

    def load_model(self, model):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.model_graph = tf.Graph()
        with tf.Session(graph=self.model_graph, config=config) as sess:
            self.session = sess
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def predict(self, image, threshold=0.5):
        # definite input and output tensors for detection graph
        image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')

        # each box represents a part of the image where a particular object was detected
        detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')

        # score represent the probability of each class
        detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')

        image = self.process_img(image)

        # actual detection
        (boxes, scores, classes) = self.session.run([detection_boxes, detection_scores, detection_classes],
                                                    feed_dict={image_tensor: np.expand_dims(image, axis=0)})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        for i, box in enumerate(boxes):
            if scores[i] > threshold:
                traffic_light_id = self.classes[classes[i]]

                if DEBUG:
                    self.detected_img(image, traffic_light_id)

                return traffic_light_id, scores[i]
            else:
                self.detected_img(image, TrafficLight.UNKNOWN)

        return None, None

    def process_img(self, image):
        image_w , image_h = image.shape[:2]
        size_max = 300

        if image_h > size_max or image_w > size_max:
            scaling_factor = size_max / float(image_h)
            if scaling_factor > size_max / float(image_w):
                scaling_factor = size_max / float(image_w)
            # resize
            image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def detected_img(self, image, traffic_light_id):
        detected_img_site = os.path.dirname(os.path.realpath(__file__)) + '/../../../../detected_img/site/'
        detected_img_sim = os.path.dirname(os.path.realpath(__file__)) + '/../../../../detected_img/sim/'

        # reconvert the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.config['is_site']:
            cv2.imwrite(os.path.join(detected_img_site, "image_%04i_%d.jpg" % (self.img_counter, traffic_light_id)),
                        image)
        else:
            cv2.imwrite(os.path.join(detected_img_sim, "image_%04i_%d.jpg" % (self.img_counter, traffic_light_id)),
                        image)
        self.img_counter += 1
