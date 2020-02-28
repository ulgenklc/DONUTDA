#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from itertools import compress
import numpy as np
import eztda as et
import os
import cv2
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread, imsave
from skimage import exposure, img_as_float


# In[4]:


class EZTDA_GUI:
    def __init__(self,master):
        self.master=master
        master.title("EZ-TDA 1.0")
        
        self.alphaa = Label(master, text= 'Brightness:').place(x=370,y=10)
        self.alpha = Entry(master)
        self.alpha.insert(END,'1')
        self.alpha.place(x=370,y=30)
        
        self.betaa = Label(master, text= 'Blur:').place(x=370,y=50)
        self.beta = Entry(master)
        self.beta.insert(END,'0')
        self.beta.place(x=370,y=70)
        
        self.kerneel=Label(master,text='Select a Kernel').place(x=360,y=90)
        self.kernel=ttk.Combobox(master, values=['No Kernel','3x3 Circular Kernel','3x3 Elliptical Kernel','5x5 Square Kernel','5x5 Circular Kernel','5x5 Elliptical Kernel'],state='readonly')
        self.kernel.current(0)
        self.kernel.place(x=360,y=110)
        
        self.morphologyy=Label(master, text='Select a Morphological').place(x=360,y=130)
        self.morphoology=Label(master, text='Transformation').place(x=360,y=150)
        self.morphology= ttk.Combobox(master, values=['None','Erosion','Dilation','Opening', 'Closing','Gradient','Tophat','Blackhat'],state='readonly')
        self.morphology.current(0)
        self.morphology.place(x=360,y=170)
        
        self.equalization=Label(master, text='Brightness Equalization').place(x=360,y=190)
        self.equalization=ttk.Combobox(master, values=['None','Contrast Streching','Histogram Equalization', 'Adaptive Equalization'],state='readonly')
        self.equalization.current(0)
        self.equalization.place(x=360,y=210)
        
        
        self.var=IntVar()
        self.askROI1=Radiobutton(master, variable=self.var, text='ROIs are filled', value='1')
        self.askROI2=Radiobutton(master,variable=self.var,text='ROIs are loopy',value='2')
        self.askROI1.place(x=370,y=270)
        self.askROI2.place(x=370,y=290)
        
        #self.invert=IntVar()
        #self.invert_button=Checkbutton(master,variable=self.invert, text='Invert' )
        #self.invert_button.place(x=370,y=260)
        
        self.pre_process_image=Button(master, text='Pre-Process', command=lambda: self.pre_process(raw_img,float(self.alpha.get()),float(self.beta.get()),self.kernel.get(),self.morphology.get(),self.equalization.get()))
        self.pre_process_image.place(x=390,y=240)
        
        self.homology=Button(master, text='Find ROIs', command=lambda: self.find_ROIs(img_go))
        self.homology.place(x=397,y=320)
        
        self.canvas=Canvas(master,height=3, width=850, bg='blue')
        self.canvas.place(x=0,y=350)
        self.line=self.canvas.create_line(0,355,850,355,width=5)
        
        self.pers_threshold1=Label(master,text='Persistence Threshold:').place(x=10,y=365)
        self.pers_threshold=Scale(master, from_=0,to_=255, orient=HORIZONTAL, width=10, resolution=0.5,length=200)
        self.pers_threshold.place(x=10,y=384)
        
        self.autovar=IntVar()
        self.auto_threshold_button=Checkbutton(master,variable=self.autovar, text='Auto-Threshold' )
        self.auto_threshold_button.place(x=190,y=355)
        
        self.threshold_button=Button(master,text='Threshold', command=lambda: self.good_cycle(h_pairs, imcomp, h1_cycles), height=2, width=9)
        self.threshold_button.place(x=225,y=380)
        
        self.circul=Label(master, text='Cell Circularity:').place(x=320,y=365)
        self.circul=ttk.Combobox(master, values=['0','1','2','3','4','5','6','7','8','9','10'],state='readonly', width=3)
        self.circul.current(0)
        self.circul.place(x=345,y=385)
        
        self.convex=Label(master,text='Cell Convexity:').place(x=450,y=365)
        self.convex=ttk.Combobox(master, values=['0','1','2','3','4','5','6','7','8','9','10'],state='readonly',width=3)
        self.convex.current(0)
        self.convex.place(x=475,y=385)
        
        self.max_area=Label(master, text='Cells are at most this big:').place(x=555,y=365)
        self.max_area=Entry(master)
        self.max_area.insert(END,'1000')
        self.max_area.place(x=560,y=385)
        
        self.min_area=Label(master, text='Cells are at least this big:').place(x=705,y=365)
        self.min_area=Entry(master)
        self.min_area.insert(END,'10')
        self.min_area.place(x=710,y=385)
        
        self.canvas=Canvas(master,height=3, width=850, bg='blue')
        self.canvas.place(x=0,y=420)
        self.line=self.canvas.create_line(0,425,850,425,width=5)
        
        self.canvas=Canvas(master,height=70, width=3, bg='blue')
        self.canvas.place(x=300,y=350)
        self.line=self.canvas.create_line(300,350,300,420,width=5)
        
        self.clean=Button(master, text='Clean',command=lambda: self.clean_cycles(imcomp,good_cycles,good_pairs))
        self.clean.place(x=405, y=450)
        
        self.draw=Button(master, text='Draw',command=lambda: self.draw_cycles(imcomp,good_pairs))
        self.draw.place(x=405, y=500)
        
        self.individual = Button(master, text="Look up ROIs", command=lambda:self.create_window())
        self.individual.place(x=385,y=550)
        
        self.save_button_image=Button(master, text='Save Image',command=lambda: self.save_file(img_final))
        self.save_button_image.place(x=392, y=600)
        self.save_button_image.config(height=5,width=8)
        
        self.save_button_masks= Button(master,text='Save Masks', command=self.save_masks)
        self.save_button_masks.place(x=392,y=700)
        self.save_button_masks.config(height=5,width=8)
        

        
        self.label1=Label(master, text='Original Image').place(x=130,y=0)
        self.label2=Label(master, text='Pre-Processed Image').place(x=630,y=0)
        self.label3=Label(master, text='Raw Parcellation').place(x=130,y=425)
        self.label4=Label(master, text='Final Contours').place(x=630,y=425)
        
    def draw_cycles(self,imcomp,good_pairs):
        global contours,final_image, img_final,points,mask_false,thicknes
        size=raw_img.shape[0]*raw_img.shape[1]
        if size<62500:thicknes=1
        elif size>=62500 and size<250000:thicknes=2
        else:thicknes=4    
        if str(self.var.get())=='1' and threshold==True:
            points=imcomp.pairs_to_location(good_pairs)
            if check == False: mask_false=np.ones(len(points),dtype=bool)##assumes everything is true positive at the beginning
            img_final=cv2.cvtColor(raw_img,cv2.COLOR_GRAY2RGB)
            for i in range(len(points[:,2][mask_false])):
                cv2.circle(img_final,(points[:,2][mask_false][i].astype('int32'),points[:,1][mask_false][i].astype('int32')), 3, (60,255,20), thicknes)
            img_final1 = Image.fromarray(img_final)
            img_final1 = img_final1.resize((325, 325), Image.ANTIALIAS)
            img_final2 = ImageTk.PhotoImage(img_final1)
            if final_image is None:
                final_image= Label(image=img_final2)
                final_image.image = img_final2
                final_image.place(x=510,y=440)
            else: final_image.configure(image=img_final2); final_image.image=img_final2
        
        elif str(self.var.get())=='2' and threshold==True:
            circular=float(self.circul.get())*0.1
            convexity=float(self.convex.get())*0.1
            contours=imcomp.to_contour(good_pairs,masks,circ=circular, conv=convexity, A_high=float(self.max_area.get()), A_low=float(self.min_area.get()))    
            if check== False: mask_false=np.ones(len(contours),dtype=bool)##assumes everything is true positive at the beginning
            img_final=imcomp.draw_contour(contours[mask_false],raw_img,thickness=thicknes)
            img_final1 = Image.fromarray(img_final)
            img_final1 = img_final1.resize((325, 325), Image.ANTIALIAS)
            img_final2 = ImageTk.PhotoImage(img_final1)
            if final_image is None:
                final_image= Label(image=img_final2)
                final_image.image = img_final2
                final_image.place(x=510,y=440)
            else: final_image.configure(image=img_final2); final_image.image=img_final2    
        else:self.warning=messagebox.showwarning('Nope','Apply a Thresholding')
            
    def clean_cycles(self,imcomp,good_cycles,good_pairs):
        global masks
        if str(self.var.get())=='1' and threshold==True: messagebox.showinfo("EZ-TDA", "Nothing to clean, go to draw")
        elif str(self.var.get())=='2' and threshold==True:good_cycles_cleaned,mask_good_clean,masks=imcomp.remove_non_boundary(good_cycles)
           
    def save_masks(self):
        file = filedialog.asksaveasfile(mode='w', filetypes=(("TIF file","*.tif"),("All Files", "*.*") ))
        if str(self.var.get())=='1':
            if file:
                abs_path = os.path.abspath(file.name)
                for i in range(len(points[:,2][mask_false])):
                    mask = np.zeros(raw_img.shape, dtype=np.uint8)
                    cv2.circle(mask,(points[:,2][mask_false][i].astype('int32'),points[:,1][mask_false][i].astype('int32')), 5, (60,255,20), -1)
                    imsave(abs_path+'_'+ str(i) + '.tif', mask)
        elif str(self.var.get())=='2':
            if file:
                abs_path = os.path.abspath(file.name)
                for i,c in enumerate(contours[mask_false]):
                    mask = np.zeros(raw_img.shape, dtype=np.uint8)
                    hull = cv2.convexHull(contours[mask_false][i])
                    cv2.fillPoly(mask,[hull],255)
                    imsave(abs_path+'_'+ str(i) + '.tif', mask)
            
    def save_file(self,img_final):
        file = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
        if file:
            abs_path = os.path.abspath(file.name)
            img_final1 = Image.fromarray(img_final)
            img_final1.save(abs_path) # saves the image to the input file name. 
    
    
    def find_ROIs(self,img):
        global h_pairs, h1_cycles, imcomp
        if str(self.var.get())=='1':
            imcomp=et.ImageFiltration(img)
            pairs=et.twist_persistence(imcomp)
            h_pairs=imcomp.finite_pairs(pairs,dim=0)
            h1_cycles=None    
        elif str(self.var.get())=='2':
            imcomp = et.ImageComplex(img)
            pairs,cycles=  imcomp.persistence()
            h_pairs, h1_cycles = et.homology(pairs,cycles,1)
        else:self.warning=messagebox.showwarning('Nope','You have to specify the ROI type to proceed')
    
    def auto_threshold(self,imcomp,pairs):
        if str(self.var.get())=='1':
            pers= imcomp.persistence(pairs,mode='degree',norm=True)
            pers_ordered=np.sort(pers)
            pers_pad=np.pad(pers_ordered,(1,1),'constant')
            der_2=pers_pad[2:]+pers_pad[:-2]-2*pers_ordered
            i=np.argmax(der_2)
            thresh=pers[i]
        elif str(self.var.get())=='2':
            deg_pairs = imcomp.to_degree(pairs)
            sig_noise=(np.transpose(deg_pairs)[2]-np.transpose(deg_pairs)[1])
            k=len(sig_noise)/10
            signal=np.sort(sig_noise)
            thresh=signal[-int(k):][0]
        return(thresh)

    
    def select_image(self):
        global original, raw_img
        path=filedialog.askopenfilename()
        #file=os.path.basename(path)
        path = os.path.normpath(path)
        direct=path.split(os.sep)
        path_name=str()
        found=False
        for i in range(len(direct)-1):
            if direct[i]=='data_images':found=True;
            if found==True:path_name=path_name+'/' + direct[i+1]
        if len(path)>0:
            raw_img= cv2.imread('../DONUT/data_images' + '%s'%path_name,0) 
            raw_img= rgb2gray(raw_img)
            img=np.array(raw_img, dtype=np.uint8)
            img1 = Image.fromarray(img)
            img1=img1.resize((325, 325), Image.ANTIALIAS)
            img2 = ImageTk.PhotoImage(img1)
        if original is None:
            original = Label(image=img2)
            original.image = img2
            original.place(x=20, y=20)
        else:original.configure(image=img2); original.image=img2

            
    def pre_process(self,image,alpha,beta,kernel,morph,equal):
        global pre_processed, img_go
        
        if equal=='None':enhenced_img=image
        elif equal=='Contrast Streching':enhenced_img=self.enhence_image(image,alpha,beta)
        elif equal=='Histogram Equalization':enhenced_img=exposure.equalize_hist(image);enhenced_img=enhenced_img*255
        elif equal=='Adaptive Equalization':enhenced_img= exposure.equalize_adapthist(image, clip_limit=0.03);enhenced_img=enhenced_img*255
        
        if kernel=='No Kernel':
            kernel_array=np.array([1],dtype=np.uint8)
        elif kernel=='3x3 Circular Kernel':
            kernel_array=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.uint8)
        elif kernel=='3x3 Elliptical Kernel':
            kernel_array=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
        elif kernel=='5x5 Square Kernel':
            kernel_array=np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],dtype=np.uint8)
        elif kernel=='5x5 Circular Kernel':
            kernel_array=np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=np.uint8)
        elif kernel=='5x5 Elliptical Kernel':
            kernel_array=np.array([[0,0,1,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,1,0,0]],dtype=np.uint8)
        
        if morph=='None':img_go=enhenced_img
        elif morph=='Erosion': erosion = cv2.erode(enhenced_img,kernel_array,iterations = 1);img_go=erosion
        elif morph=='Dilation': dilation = cv2.dilate(enhenced_img,kernel_array,iterations = 1); img_go=dilation
        elif morph=='Opening': opening = cv2.morphologyEx(enhenced_img, cv2.MORPH_OPEN, kernel_array); img_go=opening
        elif morph=='Closing':closing = cv2.morphologyEx(enhenced_img, cv2.MORPH_CLOSE, kernel_array); img_go=closing
        elif morph=='Gradient':gradient = cv2.morphologyEx(enhenced_img, cv2.MORPH_GRADIENT, kernel_array); img_go=gradient
        elif morph=='Tophat':tophat = cv2.morphologyEx(enhenced_img, cv2.MORPH_TOPHAT, kernel_array); img_go=tophat
        elif morph=='Blackhat': blackhat = cv2.morphologyEx(enhenced_img, cv2.MORPH_BLACKHAT, kernel_array);img_go=blackhat
        
        
        enhenced_img1 = Image.fromarray(img_go)
        enhenced_img1 = enhenced_img1.resize((325, 325), Image.ANTIALIAS)
        enhenced_img2 = ImageTk.PhotoImage(enhenced_img1)
        if pre_processed is None:
            pre_processed = Label(image=enhenced_img2)
            pre_processed.image = enhenced_img2
            pre_processed.place(x=510,y=20)
        else:pre_processed.configure(image=enhenced_img2); pre_processed.image=enhenced_img2
        
    def enhence_image(self,image,alpha,beta):
        img= image.astype(np.float)
        img  = img*alpha + beta
        np.clip(img, 0, 255, img)
        return(img)
    
    def good_cycle(self,h_pairs,imcomp,h1_cycles):
        global thresholded, good_cycles, good_pairs, threshold
        threshold=True
        masks=None
        if str(self.autovar.get())=='1': thresh=self.auto_threshold(imcomp,h_pairs)
        elif str(self.autovar.get())=='0':thresh=self.pers_threshold.get()
        if str(self.var.get())=='1':
            pers= imcomp.persistence(h_pairs,mode='degree',norm=True)
            pers=pers*255
            good_cycles=None
            good_pairs=h_pairs[pers>thresh]
            points=imcomp.pairs_to_location(good_pairs)
            mask_good = np.zeros((raw_img.shape[0], raw_img.shape[1], 4))
            mask_good[points[:,1].astype('int32'), points[:,2].astype('int32'),0] = 1
            mask_good[points[:,1].astype('int32'), points[:,2].astype('int32'),3] = 1
            mask_good1 = Image.fromarray(np.uint8(mask_good[:,:,0]*255),mode='L')
            mask_good1 = mask_good1.resize((325, 325), Image.ANTIALIAS)
            mask_good2 = ImageTk.PhotoImage(mask_good1)
            if thresholded is None:
                thresholded = Label(image=mask_good2)
                thresholded.image = mask_good2
                thresholded.place(x=20, y=440)
            else:thresholded.configure(image=mask_good2); thresholded.image=mask_good2
        elif str(self.var.get())=='2':
            deg_pairs = imcomp.to_degree(h_pairs)
            good_cycles = h1_cycles[deg_pairs[:,2]-deg_pairs[:,1] > thresh]
            good_pairs = imcomp.to_degree(h_pairs[deg_pairs[:,2]-deg_pairs[:,1]>thresh])
            mask_good=imcomp.overlay(good_cycles)
            mask_good1 = Image.fromarray(np.uint8(mask_good[:,:,0]*255),mode='L')
            mask_good1 = mask_good1.resize((325, 325), Image.ANTIALIAS)
            mask_good2 = ImageTk.PhotoImage(mask_good1)
            if thresholded is None:
                thresholded = Label(image=mask_good2)
                thresholded.image = mask_good2
                thresholded.place(x=20, y=440)
            else:thresholded.configure(image=mask_good2); thresholded.image=mask_good2
        
    def helpbox(self):messagebox.showinfo("EZ-TDA", "Contact to bengieru@buffalo.edu for help")
   
    def create_window(self):
        global cycle_selection
        
        ##creates the new window
        window = Toplevel()
        window.geometry("700x400")
        
        ##creates the image panel in the new window
        img = np.array(raw_img, dtype=np.uint8)
        img1 = Image.fromarray(img)
        img1 = img1.resize((325,325), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img1)
        cycle_selection = Label(window,image=img2)
        cycle_selection.image = img2
        cycle_selection.place(x=300,y=40)
        
        #creates listbox in the new window showing the cycles(ROIS)
        Lb1=Listbox(window,selectmode=EXTENDED)
        Lb1.place(x=20,y=100)
        if str(self.var.get())=='1':items=points
        elif str(self.var.get())=='2':items=contours
        self.repopulate(Lb1,items)    
        
        ##binding for displaying the selected item(cycle) in the panel at the new window
        Lb1.bind('<<ListboxSelect>>', lambda e: self.update(Lb1,items))
        
        ##Buttons
        true_positive=Button(window,text='True Positive', bg='green', command=lambda:self.mark_cycle(Lb1,mask_false,True))
        true_positive.place(x=190,y=190)
        
        false_positive=Button(window,text='False Positive',bg='red', command=lambda:self.mark_cycle(Lb1,mask_false,False))
        false_positive.place(x=188,y=120)
        
        okkay=Button(window,text='Okay',command=window.destroy)
        okkay.place(x=205,y=290)
        
    def update(self,listbox,items):#helper function to display individual cycles
        if str(self.var.get())=='1':
            img_single=cv2.cvtColor(raw_img,cv2.COLOR_GRAY2RGB)
            cv2.circle(img_single,(items[:,2][listbox.curselection()[0]].astype('int32'),items[:,1][listbox.curselection()[0]].astype('int32')), 3, (60,255,20), thicknes)
            img_single1 = Image.fromarray(img_single)
            img_single1 = img_single1.resize((325, 325), Image.ANTIALIAS)
            img_single2 = ImageTk.PhotoImage(img_single1)
            
        elif str(self.var.get())=='2':
            img_single=imcomp.draw_contour(items,raw_img,index=listbox.curselection()[0],thickness=thicknes)
            img_single1 = Image.fromarray(img_single)
            img_single1 = img_single1.resize((325, 325), Image.ANTIALIAS)
            img_single2 = ImageTk.PhotoImage(img_single1)
        
        cycle_selection.configure(image=img_single2);cycle_selection.image=img_single2
    
    def mark_cycle(self,listbox,mask,value):##helper function to mark cycles as true/false positive
        global check
        if value==False:
            mask_false[[listbox.curselection()[0]]]=False
            listbox.itemconfig([listbox.curselection()[0]],bg='red')
        elif value==True:
            mask_false[[listbox.curselection()[0]]]=True
            listbox.itemconfig([listbox.curselection()[0]],bg='green')

        
        check=True
                
        img = np.array(raw_img, dtype=np.uint8)
        img1 = Image.fromarray(img)
        img1 = img1.resize((325,325), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img1)
        
        cycle_selection.configure(image=img2);cycle_selection.image=img2
    
    def repopulate(self,listbox,items):#helper function to populate the ROI listbox
        listbox.delete(0,END)
        for i,m in enumerate(items):
            listbox.insert(i,"ROI id %d"%i)
            if mask_false[i]==1: listbox.itemconfig(i,bg='green')
            elif mask_false[i]==0:listbox.itemconfig(i,bg='red')    


# In[5]:


root = Tk()
root.resizable(width=False, height=False)
eztda=EZTDA_GUI(root)
root.geometry('850x850+10+10')
#insert a menubar on the main window
menubar= Menu(root)
root.iconbitmap(r'C:\Users\ulgen\OneDrive\Masaüstü\DONUT\trefoil.ico')
root.config(menu=menubar)

#create a menu button labeled file that brings up menu
filemenu=Menu(menubar,tearoff=False)
menubar.add_cascade(label='Menu', menu=filemenu)

# Create entries in the "File" menu
filemenu.add_command(label='New File', command=eztda.select_image )
filemenu.add_separator(  )
filemenu.add_command(label='Quit', command=root.destroy)

aboutmenu=Menu(menubar,tearoff=False)
menubar.add_cascade(label='About', menu=aboutmenu)
aboutmenu.add_command(label='Help',command=eztda.helpbox)

original=None
pre_processed=None
thresholded=None
final_image=None
cycle_selection=None
threshold=False
check=False

root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




