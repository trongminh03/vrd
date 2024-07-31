from torch.utils.data import Dataset
import copy
import os
import numpy as np
from collections import defaultdict
import pickle
import h5py
import json
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode

class VG_Dataset(Dataset):
    def __init__(self, type, split, train, pipeline, per_class_dataset, bgnn, clipped, filter_empty_relations, filter_non_overlap, 
                undersample_param=0.7, oversample_param=0.07, images='./data/datasets/VG/VG_100K/', 
                mapping_dictionary='./data/datasets/VG/VG-SGG-dicts-with-attri.json', image_data='./data/datasets/VG/image_data.json',
                vg_attribute_h5='./data/datasets/VG/VG-SGG-with-attri.h5'):      
        self.type = type
        self.split = split
        self.train = train
        self.pipeline = pipeline
        self.per_class_dataset = per_class_dataset
        self.bgnn = bgnn
        self.clipped = clipped
        self.filter_non_overlap = filter_non_overlap
        self.filter_empty_relations = filter_empty_relations
        self.images = images
        self.mapping_dictionary = mapping_dictionary
        self.image_data = image_data
        self.VG_attribute_h5 = vg_attribute_h5
        self.number_of_validation_images = 5000
        self.box_scale = 1024
        
        if split == 'train':
            self.mask_location = ""
        elif split == 'val':
            self.mask_location = ""
        else:
            self.mask_location = ""

        self.mask_exists = os.path.isfile(self.mask_location)
        self.clamped = True if "clamped" in self.mask_location else ""
        self.per_class_dataset = per_class_dataset if split == 'train' else False
        self.bgnn = bgnn if split == 'train' else False
        self.clipped = clipped
        self.precompute = False if (filter_empty_relations or filter_non_overlap) else True
        
        self.data_list = self.load_data_list()
        self.len_data_list = len(self.data_list) 


        try:
            with open('./data/datasets/images_to_remove.txt', 'r') as f:
                ids = f.readlines()
            self.ids_to_remove = {int(x.strip()) : 1 for x in ids[0].replace('[', '').replace(']','').split(",")}
        except:
            self.ids_to_remove = []
        # self._process_data()
        self.dataset_dicts = self._fetch_data_dict()
        self.register_dataset()
        try:
            statistics = self.get_statistics()
        except:
            pass
        if self.bgnn:
            freq = statistics['fg_rel_count'] / statistics['fg_rel_count'].sum() 
            freq = freq.numpy()
            # oversample_param = cfg.DATASETS.VISUAL_GENOME.OVERSAMPLE_PARAM
            # undersample_param = cfg.DATASETS.VISUAL_GENOME.UNDERSAMPLE_PARAM
            oversampling_ratio = np.maximum(np.sqrt((oversample_param / (freq + 1e-5))), np.ones_like(freq))[:-1]
            sampled_dataset_dicts = []
            sampled_num = []
            unique_relation_ratios = []
            unique_relations_dict = []
            for record in self.dataset_dicts:
                relations = record['relations']
                if len(relations) > 0:
                    unique_relations = np.unique(relations[:,2])
                    repeat_num = int(np.ceil(np.max(oversampling_ratio[unique_relations])))
                    for rep_idx in range(repeat_num):
                        sampled_num.append(repeat_num)
                        unique_relation_ratios.append(oversampling_ratio[unique_relations])
                        sampled_dataset_dicts.append(record)
                        unique_relations_dict.append({rel:idx for idx, rel in enumerate(unique_relations)})
                else:
                    sampled_dataset_dicts.append(record)
                    sampled_num.append(1)
                    unique_relation_ratios.append([])
                    unique_relations_dict.append({})

            self.dataset_dicts = sampled_dataset_dicts
            self.dataloader = BGNNSampler(self.dataset_dicts, sampled_num, oversampling_ratio, undersample_param, unique_relation_ratios, unique_relations_dict)
            # DatasetCatalog.remove('VG_{}'.format(self.split))
            self.register_dataset(dataloader=True)
            # MetadataCatalog.get('VG_{}'.format(self.split)).set(statistics=statistics) 
            print (self.idx_to_predicates, statistics['fg_rel_count'].numpy().tolist())

        if self.per_class_dataset:
            freq = statistics['fg_rel_count'] / statistics['fg_rel_count'].sum() 
            freq = freq.numpy()
            # oversample_param = cfg.DATASETS.VISUAL_GENOME.OVERSAMPLE_PARAM
            # undersample_param = cfg.DATASETS.VISUAL_GENOME.UNDERSAMPLE_PARAM
            oversampling_ratio = np.maximum(np.sqrt((oversample_param / (freq + 1e-5))), np.ones_like(freq))[:-1]
            unique_relation_ratios = defaultdict(list)
            unique_relations_dict = defaultdict(list)     
            per_class_dataset = defaultdict(list)
            sampled_num = defaultdict(list)
            for record in self.dataset_dicts:
                relations = record['relations']
                if len(relations) > 0:
                    unique_relations = np.unique(relations[:,2])
                    repeat_num = int(np.ceil(np.max(oversampling_ratio[unique_relations])))
                    for rel in unique_relations:
                        per_class_dataset[rel].append(record)   
                        sampled_num[rel].append(repeat_num)
                        unique_relation_ratios[rel].append(oversampling_ratio[unique_relations]) 
                        unique_relations_dict[rel].append({rel:idx for idx, rel in enumerate(unique_relations)})
            self.dataloader = ClassBalancedSampler(per_class_dataset, len(self.dataset_dicts), sampled_num, oversampling_ratio, undersample_param, unique_relation_ratios, unique_relations_dict)
            # DatasetCatalog.remove('VG_{}'.format(self.split))
            self.register_dataset(dataloader=True)
            # MetadataCatalog.get('VG_{}'.format(self.split)).set(statistics=statistics) 
            print (self.idx_to_predicates, statistics['fg_rel_count'].numpy().tolist())
        # import IPython; IPython.embed()

        # if self.train == True:
        #     for i in range(self.threshold_ram):
        #         print("Preparing data: {}/{}".format(i, self.threshold_ram))
        #         self.data_list[i] = self.prepare_data(i)

    def __getitem__(self, idx: int) -> dict:
        idx = idx % self.len_data_list 
        
        # if self.train == False or idx >= self.threshold_ram:
        #     data = self.prepare_data(idx)
        #     if data is None:
        #         raise Exception(f'Cannot find valid image path: {idx}.'
        #                     'Please check your image path and pipeline')
        #     return data
        # else:
        #     return self.data_list[idx]
        
        data = self.prepare_data(idx)
        if data is None:
            raise Exception(f'Cannot find valid image path: {idx}.'
                        'Please check your image path and pipeline')
        if self.save_idx: 
            return data, idx
        return data
    
    def __len__(self) -> int:
        return self.len_data_list  * self.number_repeats

    def _fetch_data_dict(self):
        """
        Load data in detectron format
        """
        fileName = "tmp/visual_genome_{}_data_{}{}{}{}{}{}{}{}.pkl".format(self.split, 'masks' if self.mask_exists else '', '_oi' if 'oi' in self.mask_location else '', "_clamped" if self.clamped else "", "_precomp" if self.precompute else "", "_clipped" if self.clipped else "", '_overlapfalse' if not self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP else "", '_emptyfalse' if not self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS else '', "_perclass" if self.per_class_dataset else '')
        print("Loading file: ", fileName)
        if os.path.isfile(fileName):
            #If data has been processed earlier, load that to save time
            with open(fileName, 'rb') as inputFile:
                dataset_dicts = pickle.load(inputFile)
        else:
            #Process data
            os.makedirs('tmp', exist_ok=True)
            dataset_dicts = self._process_data()
            with open(fileName, 'wb') as inputFile:
                pickle.dump(dataset_dicts, inputFile)
        return dataset_dicts
    
    def _process_data(self):
        self.VG_attribute_h5 = h5py.File(self.VG_attribute_h5, 'r')
        
        # Remove corrupted images
        image_data = json.load(open(self.image_data, 'r'))
        self.corrupted_ims = ['1592', '1722', '4616', '4617']
        self.image_data = []
        for i, img in enumerate(image_data):
            if str(img['image_id']) in self.corrupted_ims:
                continue
            self.image_data.append(img)
        assert(len(self.image_data) == 108073)
        self.masks = None
        if self.mask_location != "":
            try:
                with open(self.mask_location, 'rb') as f:
                    self.masks = pickle.load(f)
            except:
                pass
        dataset_dicts = self._load_graphs()
        return dataset_dicts

    def _load_graphs(self):
        """
        Parse examples and create dictionaries
        """
        data_split = self.VG_attribute_h5['split'][:]
        split_flag = 2 if self.split == 'test' else 0
        split_mask = data_split == split_flag
        
        #Filter images without bounding boxes
        split_mask &= self.VG_attribute_h5['img_to_first_box'][:] >= 0
        if self.filter_empty_relations:
            split_mask &= self.VG_attribute_h5['img_to_first_rel'][:] >= 0
        image_index = np.where(split_mask)[0]
        
        if self.split == 'val':
            image_index = image_index[:self.number_of_validation_images]
        elif self.split == 'train':
            image_index = image_index[self.number_of_validation_images:]
        
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True
        
        # Get box information
        all_labels = self.VG_attribute_h5['labels'][:, 0]
        all_attributes = self.VG_attribute_h5['attributes'][:, :]
        all_boxes = self.VG_attribute_h5['boxes_{}'.format(self.box_scale)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box
        
        # Convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]
        
        first_box_index = self.VG_attribute_h5['img_to_first_box'][split_mask]
        last_box_index = self.VG_attribute_h5['img_to_last_box'][split_mask]
        first_relation_index = self.VG_attribute_h5['img_to_first_rel'][split_mask]
        last_relation_index = self.VG_attribute_h5['img_to_last_rel'][split_mask]

        #Load relation labels
        all_relations = self.VG_attribute_h5['relationships'][:]
        all_relation_predicates = self.VG_attribute_h5['predicates'][:, 0]
        
        image_indexer = np.arange(len(self.image_data))[split_mask]
        # Iterate over images
        dataset_dicts = []
        num_rels = []
        num_objs = []
        for idx, _ in enumerate(image_index):
            record = {}
            #Get image metadata
            image_data = self.image_data[image_indexer[idx]]
            record['file_name'] = os.path.join(self.images, '{}.jpg'.format(image_data['image_id']))
            record['image_id'] = image_data['image_id']
            record['height'] = image_data['height']
            record['width'] = image_data['width']
            if self.clipped:
                if image_data['coco_id'] in self.ids_to_remove:
                    continue
            #Get annotations
            boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
            gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]
            gt_attributes = all_attributes[first_box_index[idx]:last_box_index[idx] + 1, :]

            if first_relation_index[idx] > -1:
                predicates = all_relation_predicates[first_relation_index[idx]:last_relation_index[idx] + 1]
                objects = all_relations[first_relation_index[idx]:last_relation_index[idx] + 1] - first_box_index[idx]
                predicates = predicates - 1
                relations = np.column_stack((objects, predicates))
            else:
                assert not self.filter_empty_relations
                relations = np.zeros((0, 3), dtype=np.int32)
            
            if self.filter_non_overlap and self.split == 'train':
                # Remove boxes that don't overlap
                boxes_list = Boxes(boxes)
                ious = pairwise_iou(boxes_list, boxes_list)
                relation_boxes_ious = ious[relations[:,0], relations[:,1]]
                iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
                if iou_indexes.size > 0:
                    relations = relations[iou_indexes]
                else:
                    #Ignore image
                    continue
            #Get masks if possible
            if self.masks is not None:
                try:
                    gt_masks = self.masks[image_data['image_id']]
                except:
                    print (image_data['image_id'])
            record['relations'] = relations
            objects = []
            # if len(boxes) != len(gt_masks):
            mask_idx = 0
            for obj_idx in range(len(boxes)):
                resized_box = boxes[obj_idx] / self.box_scale * max(record['height'], record['width'])
                obj = {
                      "bbox": resized_box.tolist(),
                      "bbox_mode": BoxMode.XYXY_ABS,
                      "category_id": gt_classes[obj_idx] - 1,
                      "attribute": gt_attributes[obj_idx],           
                }
                if self.masks is not None:
                    if gt_masks['empty_index'][obj_idx]:
                        refined_poly = []
                        for poly_idx, poly in enumerate(gt_masks['polygons'][mask_idx]):
                            if len(poly) >= 6:
                                refined_poly.append(poly)
                        obj["segmentation"] = refined_poly
                        mask_idx += 1
                    else:
                        obj["segmentation"] = []
                    if len(obj["segmentation"]) > 0:
                        objects.append(obj)
                else:
                    objects.append(obj)
            num_objs.append(len(objects))
            num_rels.append(len(relations))  
            
            record['annotations'] = objects
            dataset_dicts.append(record)
        print ("Max Rels:", np.max(num_rels), "Max Objs:", np.max(num_objs))
        print ("Avg Rels:", np.mean(num_rels), "Avg Objs:", np.mean(num_objs))
        print ("Median Rels:", np.median(num_rels), "Median Objs:", np.median(num_objs))
        return dataset_dicts
    
    # def load_data_list(self):
    #     """Load annotation file to get skeleton information."""
    #     assert self.ann_file.endswith('.pkl')
    #     data_list = load_pkl(self.ann_file)

    #     split, annos = data_list['split'], data_list['annotations']
    #     identifier = 'filename' if 'filename' in annos[0] else 'frame_dir'
    #     split = set(split[self.split])
    #     data_list = [x for x in annos if x[identifier] in split]   
    #     return data_list

    # def prepare_data(self, idx):
    #     """Get data processed by ``self.pipeline``.

    #     Args:
    #         idx (int): The index of ``data_info``.

    #     Returns:
    #         dict: The result dict contains the following keys:
    #     """
    #     data_info = copy.deepcopy(self.data_list[idx])
    #     data_info2 = copy.deepcopy(self.data_list[idx])
    #     return self.process(data_info), self.process(data_info2)

    # def process(self, data_info):
    #     for transform_component in self.pipeline:
    #         data_info = transform_component.transform(data_info)
    #     return data_info
        
