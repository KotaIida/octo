import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py

def extract_img_from_hdf(hdf_file, step, img_key = "top"):
    with h5py.File(hdf_file, 'r') as hdf_f:
        img = hdf_f["observations"]["images"][img_key][:].copy()[step]
 
    return img


class GoalImgCreator:
    FIG_WIDTH = 16
    FIG_HEIGHT = 10
    FIG_HEIGHT_PER_ROW = 5
    CAND_SHOW_COLS = 5
    FONTSIZE = 20
    EDGE_COLOR = (0,255,0)
    EDGE_THICK = 2
    AREA_TH = 150
    
    def __init__(self, init_img, bg_img, final_imgs):
        self.init_img = init_img
        self.bg_img = bg_img
        self.init_fg_img = cv2.subtract(init_img, bg_img)
        self.init_fg_img_gray = cv2.cvtColor(self.init_fg_img, cv2.COLOR_RGB2GRAY)    
        self.init_fg_ctrs, _ = cv2.findContours(self.init_fg_img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        self.final_imgs = final_imgs
        self.final_fg_imgs = [cv2.subtract(final_img, bg_img) for final_img in final_imgs]
        self.final_fg_imgs_gray = [cv2.cvtColor(final_fg_img, cv2.COLOR_RGB2GRAY) for final_fg_img in self.final_fg_imgs]    
        self.final_fg_ctrs_list = [cv2.findContours(final_fg_img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] for final_fg_img_gray in self.final_fg_imgs_gray]
        self.info = {}
        self.info["del"] = {}
        self.info["repl"] = {}
        self.info["repl"]["src"] = {}
        self.info["repl"]["new"] = {}
        
    def setup(self, del_target_indices=None, repl_src_indices=None, repl_new_indices_list=None):
        self.register_del_targets(del_target_indices)
        self.register_replace_targets(repl_src_indices, repl_new_indices_list)

        self.show_del_targets()
        self.show_repl_targets()


    def register_del_targets(self, del_target_indices=None):                        
        del_cands = []    
        del_masks = []
        plt.figure(figsize=(self.FIG_WIDTH, (len(self.init_fg_ctrs)//self.CAND_SHOW_COLS+1)*self.FIG_HEIGHT_PER_ROW))
        for i, init_fg_ctr in enumerate(self.init_fg_ctrs):
            x,y,w,h = cv2.boundingRect(init_fg_ctr)

            del_mask = np.zeros_like(self.init_img[:, :, 0])
            cv2.drawContours(del_mask, [init_fg_ctr], -1, (1), thickness=cv2.FILLED)
            del_masks.append(del_mask.astype(bool))
        
            del_cand = self.init_img[y:y+h, x:x+w]
            del_cands.append(del_cand)
            
            ax = plt.subplot(len(self.init_fg_ctrs)//self.CAND_SHOW_COLS+1, self.CAND_SHOW_COLS, i+1)
            ax.imshow(del_cand)
            ax.set_title(i, fontsize=self.FONTSIZE)
        plt.show()
        
        if del_target_indices is None:
            del_target_indices = input("Input the indices of the images you wanna delete, separated by commas! > ")
            del_target_indices = [int(i) for i in del_target_indices.split(",")]
        del_target_indices.sort()


        for i, del_target_idx in enumerate(del_target_indices):
            self.info["del"][i] = {}
            self.info["del"][i]["crop"] = del_cands[del_target_idx]
            self.info["del"][i]["mask"] = del_masks[del_target_idx]


    def show_del_targets(self):
        del_target_img = self.init_img.copy()
        
        plt.figure(figsize=(self.FIG_WIDTH, (len(self.info["del"])//self.CAND_SHOW_COLS+1)*self.FIG_HEIGHT_PER_ROW))
        for del_target_idx, del_info in self.info["del"].items():                
            del_target_img[del_info["mask"]] = [255, 255, 0]
            
            ax = plt.subplot(len(self.info["del"])//self.CAND_SHOW_COLS+1, self.CAND_SHOW_COLS, del_target_idx+1)
            ax.imshow(del_info["crop"])
            ax.set_title(f"Delte Target Index: {del_target_idx}", fontsize=self.FONTSIZE)
        plt.show()
        
        plt.figure(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
        plt.title("Delete Target Area", fontsize=self.FONTSIZE)
        plt.imshow(del_target_img)
        plt.show()
        

    def register_replace_targets(self, repl_src_indices=None, repl_new_indices_list=None):
        repl_src_cands = []    
        repl_src_masks = []
        plt.figure(figsize=(self.FIG_WIDTH, (len(self.init_fg_ctrs)//self.CAND_SHOW_COLS+1)*self.FIG_HEIGHT_PER_ROW))
        for i, init_fg_ctr in enumerate(self.init_fg_ctrs):
            x,y,w,h = cv2.boundingRect(init_fg_ctr)

            repl_src_mask = np.zeros_like(self.init_img[:, :, 0])
            cv2.drawContours(repl_src_mask, [init_fg_ctr], -1, (1), thickness=cv2.FILLED)
            repl_src_masks.append(repl_src_mask.astype(bool))
        
            repl_src_cand = self.init_img[y:y+h, x:x+w]
            repl_src_cands.append(repl_src_cand)
            
            ax = plt.subplot(len(self.init_fg_ctrs)//self.CAND_SHOW_COLS+1, self.CAND_SHOW_COLS, i+1)
            ax.imshow(repl_src_cand)
            ax.set_title(i, fontsize=self.FONTSIZE)
        plt.show()
        
        if repl_src_indices is None:
            repl_src_indices = input("Input the indices of the images you wanna choose as a replace source area, separated by commas! > ")
            repl_src_indices = [int(i) for i in repl_src_indices.split(",")]
        repl_src_indices.sort()

        for i, repl_src_idx in enumerate(repl_src_indices):
            self.info["repl"]["src"][i] = {}
            self.info["repl"]["src"][i]["crop"] = repl_src_cands[repl_src_idx]
            self.info["repl"]["src"][i]["mask"] = repl_src_masks[repl_src_idx]


        repl_new_final_imgs_all = []
        repl_new_cands_all = []    
        repl_new_masks_all = []
        
        for final_img_idx, final_fg_ctrs in enumerate(self.final_fg_ctrs_list):            
            repl_new_final_imgs = []
            repl_new_cands = []    
            repl_new_masks = []
    
            
            final_img = self.final_imgs[final_img_idx]
            plt.figure(figsize=(self.FIG_WIDTH, (len(final_fg_ctrs)//self.CAND_SHOW_COLS+1)*self.FIG_HEIGHT_PER_ROW))
            for i, final_fg_ctr in enumerate(final_fg_ctrs):
                repl_new_final_imgs.append(final_img)                
                x,y,w,h = cv2.boundingRect(final_fg_ctr)
    
                repl_new_mask = np.zeros_like(final_img[:, :, 0])
                cv2.drawContours(repl_new_mask, [final_fg_ctr], -1, (1), thickness=cv2.FILLED)
                repl_new_masks.append(repl_new_mask.astype(bool))
            
                repl_new_cand = final_img[y:y+h, x:x+w]
                repl_new_cands.append(repl_new_cand)
                
                ax = plt.subplot(len(final_fg_ctrs)//self.CAND_SHOW_COLS+1, self.CAND_SHOW_COLS, i+1)
                ax.imshow(repl_new_cand)
                ax.set_title(i, fontsize=self.FONTSIZE)
            plt.show()
            
            if repl_new_indices_list is None:
                repl_new_indices = input("Input the indices of the images you wanna choose as a replace new area, separated by commas! > ")
                repl_new_indices = [int(i) for i in repl_new_indices.split(",")]
            else:
                repl_new_indices = repl_new_indices_list[final_img_idx]
            repl_new_indices.sort()
    
            for repl_new_idx in repl_new_indices:
                repl_new_final_imgs_all.append(repl_new_final_imgs[repl_new_idx])
                repl_new_cands_all.append(repl_new_cands[repl_new_idx])
                repl_new_masks_all.append(repl_new_masks[repl_new_idx])


        for i ,(repl_new_final_img, repl_new_cand, repl_new_mask) in enumerate(zip(repl_new_final_imgs_all, repl_new_cands_all, repl_new_masks_all)):
            self.info["repl"]["new"][i] = {}
            self.info["repl"]["new"][i]["final_img"] = repl_new_final_img
            self.info["repl"]["new"][i]["crop"] = repl_new_cand
            self.info["repl"]["new"][i]["mask"] = repl_new_mask


    def show_repl_targets(self):
        repl_src_img = self.init_img.copy()
        
        plt.figure(figsize=(self.FIG_WIDTH, (len(self.info["repl"]["src"])//self.CAND_SHOW_COLS+1)*self.FIG_HEIGHT_PER_ROW))
        for repl_src_idx, repl_src_info in self.info["repl"]["src"].items():                
            repl_src_img[repl_src_info["mask"]] = [255, 255, 0]
            
            ax = plt.subplot(len(self.info["repl"]["src"])//self.CAND_SHOW_COLS+1, self.CAND_SHOW_COLS, repl_src_idx+1)
            ax.imshow(repl_src_info["crop"])            
            ax.set_title(f"Replace Source Area Index: {repl_src_idx}", fontsize=self.FONTSIZE)
        plt.show()
        
        plt.figure(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
        plt.imshow(repl_src_img)
        plt.title("Replace Source Area", fontsize=self.FONTSIZE)
        plt.show()


        
        for repl_new_idx, dic in self.info["repl"]["new"].items():
            repl_new_img = dic["final_img"].copy()            
            
            plt.figure(figsize=(self.FIG_WIDTH, (len(self.info["repl"]["new"])//self.CAND_SHOW_COLS+1)*self.FIG_HEIGHT_PER_ROW))
            
            repl_new_img[dic["mask"]] = [255, 255, 0]
            
            plt.imshow(dic["crop"])
            plt.title(f"Replace New Area Index: {repl_new_idx}", fontsize=self.FONTSIZE)
            plt.show()
            
            plt.figure(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
            plt.title("Replace New Area", fontsize=self.FONTSIZE)
            plt.imshow(repl_new_img)
            plt.show()


    def __call__(self, base_img, del_tgt_indices, repl_index_pairs):
        goal_img = base_img.copy()
    
        fg_img = cv2.subtract(base_img, self.bg_img)        
        fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_RGB2GRAY)    
        fg_ctrs, _ = cv2.findContours(fg_img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        # delete        
        for del_tgt_idx in del_tgt_indices:            
            del_tgt_mask = self.info["del"][del_tgt_idx]["mask"]
            del_tgt_similarities = []
            del_area_masks = []
            
            for i, fg_ctr in enumerate(fg_ctrs):
                area = cv2.contourArea(fg_ctr)
                if area < self.AREA_TH:
                    continue
                base_mask = np.zeros_like(base_img[:, :, 0])
                cv2.drawContours(base_mask, [fg_ctr], -1, (1), thickness=cv2.FILLED)
                similarity = self.calc_similarity(base_img, base_mask, self.init_img, del_tgt_mask)
                del_tgt_similarities.append(similarity)   
                del_area_masks.append(base_mask)
            
            max_sim_idx = np.argmax(del_tgt_similarities)
            del_area_mask = del_area_masks[max_sim_idx]
            del_area_mask = del_area_mask.astype(bool)
            goal_img[del_area_mask] = self.bg_img[del_area_mask]

        # replace
        for repl_index_pair in repl_index_pairs:            
            repl_src_idx, repl_new_idx = repl_index_pair
            repl_src_crop = self.info["repl"]["src"][repl_src_idx]["crop"]
            repl_new_info = self.info["repl"]["new"][repl_new_idx]
            repl_new_mask = repl_new_info["mask"]
            repl_new_final_img = repl_new_info["final_img"]
            
            repl_src_similarities = []
            repl_src_area_masks = []
            
            for i, fg_ctr in enumerate(fg_ctrs):
                area = cv2.contourArea(fg_ctr)
                if area < self.AREA_TH:
                    continue
                base_mask = np.zeros_like(base_img[:, :, 0])
                cv2.drawContours(base_mask, [fg_ctr], -1, (1), thickness=cv2.FILLED)
                similarity = self.calc_similarity(base_img, base_mask, repl_new_final_img, repl_new_mask)
                repl_src_similarities.append(similarity)   
                repl_src_area_masks.append(base_mask)

            
            max_sim_idx = np.argmax(repl_src_similarities)
            repl_src_area_mask = repl_src_area_masks[max_sim_idx]
            repl_src_area_mask = repl_src_area_mask.astype(bool)
            goal_img[repl_src_area_mask] = self.bg_img[repl_src_area_mask]

            repl_src_area_min_y, repl_src_area_min_x = np.array(np.where(repl_src_area_mask)).min(axis=1)
            repl_new_area_min_y, repl_new_area_min_x = np.array(np.where(repl_new_mask)).min(axis=1)            
            repl_new_area_max_y, repl_new_area_max_x = np.array(np.where(repl_new_mask)).max(axis=1)            
            repl_new_area_h = repl_new_area_max_y - repl_new_area_min_y
            repl_new_area_w = repl_new_area_max_x - repl_new_area_min_x
            
            repl_new_area_mask = np.zeros_like(repl_src_area_mask, dtype=bool)
            repl_new_area_mask[repl_src_area_min_y:repl_src_area_min_y+repl_new_area_h+1, repl_src_area_min_x:repl_src_area_min_x+repl_new_area_w+1] = repl_new_mask[repl_new_area_min_y:repl_new_area_max_y+1, repl_new_area_min_x:repl_new_area_max_x+1]

            goal_img[repl_new_area_mask] = repl_new_final_img[repl_new_mask]

        return goal_img

    
    # @classmethod
    # def calc_similarity(self, tgt_img, ref_img):
    #     tgt_img_resize = cv2.resize(tgt_img, ref_img.shape[1::-1])
    #     similarity = np.corrcoef(tgt_img_resize.flatten(), ref_img.flatten())[1, 0]

    #     return similarity

    def calc_similarity(self, tgt_img, tgt_mask, ref_img, ref_mask):
        channels = ['r', 'g', 'b']
        method = cv2.HISTCMP_CORREL
        
        similarities = []
        
        # plt.figure(figsize=(12, 6))        
        for i, channel in enumerate(channels):
            hist1 = cv2.calcHist([tgt_img], [i], tgt_mask.astype(np.uint8), [256], [0, 256])
            hist2 = cv2.calcHist([ref_img], [i], ref_mask.astype(np.uint8), [256], [0, 256])
                    
            # hist1 = cv2.normalize(hist1, hist1).flatten()
            # hist2 = cv2.normalize(hist2, hist2).flatten()
            hist1 = hist1.flatten()
            hist2 = hist2.flatten()
            
            similarity = cv2.compareHist(hist1, hist2, method)
            similarities.append(similarity)


        #     plt.subplot(2, 3, i + 1)
        #     plt.plot(hist1, color=channel, label=f'Image 1 - {channel.upper()}')
        #     plt.plot(hist2, color=channel, linestyle='dashed', label=f'Image 2 - {channel.upper()}')
        #     plt.title(f'Histogram Comparison ({channel.upper()})')
        #     plt.xlim([0, 256])
        #     plt.legend()
            
        # plt.tight_layout()
        # plt.show()
        
        return np.mean(similarities)
