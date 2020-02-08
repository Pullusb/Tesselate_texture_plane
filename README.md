# Tesselate texture plane
Blender addon - Tesselate texture plane

**[Download latest](https://github.com/Pullusb/Tesselate_texture_plane/archive/master.zip)**

<!-- ### [Demo Youtube]() -->

/!\ Can crash easily with some images and settings, be sure to save bedore
If it crash on your image, try different settings. if it still crash... well I'm sorry.

/!\ this addon need [opencv](https://pypi.org/project/opencv-python/) and [triangle](https://rufat.be/triangle/) module see below how to install

Want to support me? [Check out how](http://www.samuelbernou.fr/donate)

---  

## Description

<!-- Get some gif images to show the thing ! -->

<!-- ![example](https://github.com/Pullusb/images_repo/raw/master/blablabla.png) -->


## Introduction note

The script was made by taking code portions of this [tesselate addon within the rigging scripts of "les fée spéciales" studio](https://github.com/LesFeesSpeciales/blender-rigging-scripts)(Even if it wasn't a fork to begin with). Thanks to them I had a good base to start.

Here are the main upgrade:
- works in 2.8
- use Opencv module instead of Scikit/Skyimage (still use Triangle)
- better handling of holes when there are nested shapes and holes (might still not be perfect though).
- expose options for tweaking tesselation (This makes it less stable than original).
- Support UV that are deformed

## Needed module


This addons use two python modules to work. you have to install _triangulate_ and _opencv_.  
Be sure to download a version compatible with Blender 2.80's python 3.7.  
This means you ought to have the same python version installed and use pip install through this version. If you don't know how to do that look at "how to install module for specific python version".  

If you install through pip with `pip install opencv-python` and `pip install triangle`.  
You should see the folders of the modules in the python installation directory. 
In `<python_version>/lib/sites-packages`.  
Now copy opencv and triangle to the Blender module folder. If you don't know we're your Blender modules folders are, look at. "Were are the scripts folder in Blender". (Or video "addons and script installation in Blender", version quick or long).

Opencv should have install some library dependency that you can see listed in console at pip install (make sure to copy them as well):

About module installation :
if triangle doesn't install properly with pip, follow [this tutorial on module installation by Damien Picard](http://lacuisine.tech/blog/2017/10/19/how-to-install-python-libs-in-blender-part-1/).




### Where ?
Panel in sidebar : 3D view > sidebar 'N' > Tool > Tesselate plane


### Note

If there is any uv_project modifier in stack, those will be applied automatically before running _(maybe add a checkbox to choose to bypass...)_

---

## Todo:
- demo : make a demo

- stability : problem... crash inside Traingle module command, dont know what the problem is...

- puppet mode : make a planar snap bone creation tool to make a quick AE puppet style mode of rigging

- option to get only contour with Ngon fill (easy)

- add a pixel margin on image np.array (usefull when no alpha between borders), can be an option True by default

- for options: pix_margin and pix_clean, change pixel values to a value relative to the shortest side of the source image (percentage ?)

- check if mesh faces are co-planar
#https://blender.stackexchange.com/questions/107357/how-to-find-if-geometry-linked-to-an-edge-is-coplanar


optional todo:
- maybe put a default less dense mesh

- support messed up UVs
   - almost OK, homography with perspective not always giving accurate results with big deformation.

when transformed to operator :
- choose the texture source if multiple in shader ? (or ensure this is the one connected to surface)
- add presets ?




<!--
DONE:
- On hard to understand values map to a range of 0-1
    - simplify 0 ~ 0.002, or 0.001 ~ 0.002 with way to turn it off in interface (send 0)
    - aeration 0.0001 ~ 0.01 
    - min_angles 15 ~ 35
 -->