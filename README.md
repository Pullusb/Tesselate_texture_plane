# Tesselate texture plane
Blender addon - Tesselate texture plane  
  
**[Download latest](https://github.com/Pullusb/Tesselate_texture_plane/archive/master.zip)**
  
<!-- ### [Demo Youtube]() -->

/!\ Can crash easily with some images and settings (stable in "contour only" mode), be sure to save before use.  
If it crash on your image, try different settings. if it still crash... well I'm sorry.  

/!\ This addon need [opencv](https://pypi.org/project/opencv-python/) and [triangle](https://rufat.be/triangle/) module see below how to install

Want to support me? [Check out how](http://www.samuelbernou.fr/donate)

---  

## Description  

You can tesselate only the contour of a plane texture if you just need to strip the alpha from.
The purpose of the tesselation is to have a some vertices density in the mesh to be able to do "puppets" deformation with rigging.

Exemple:
With _import image as plane_ addon, I imported this ryu drawing image (found ramdomly on the web for test purpose).
"contour only" option generate a mesh with stripped alpha, otherwise it does the same with the contour but fill it with internal triangles.

![demo](https://github.com/Pullusb/images_repo/raw/master/tess_usage_exemple.png)


### options

Contour only : No tesselation, just mesh the contour of the shape and fill with one Ngon face per separated "island" (stable, use only open-cv module) note : dont put hole in face  

![contour only](https://github.com/Pullusb/images_repo/raw/master/tess_contour_only.png)


Simplify contour : 0=disabled, Approximation treshold of the contour polygon  
Higher value decimate the mesh contour shape  
Note: More simplification need also more pixel margin to avoid cutting though image  

![simplify](https://github.com/Pullusb/images_repo/raw/master/tess_simplify_contour_2fps.gif)


Aeration : Limit maximum tri area, low value means more density, add more triangles

Pixel margin : 0=disabled, Dilate contour around shape  

![pix margin](https://github.com/Pullusb/images_repo/raw/master/tess_pixel_margin_2fps.gif)

Pixel cleaning : Delete surface with pixel dimensions smaller than given pixel size. (Do not use if your source is pixel art)  
Usefull for cleaning rogues pixels, but carefull, pushing too much can also erode surface angles  
In the example black part have alpha at 0 (smart test image this time)  

![pix clean](https://github.com/Pullusb/images_repo/raw/master/tess_pixel_cleaning_2fps.gif)


Minimum angle : 0=disabled, "quality mesh generation" value, augment the mimimum angle limitation (note that value is not given in angle), basically means more triangle with better repartition.  
Note : This have a big influence on aeration ! Add more density variation (kind of like Dyntopo for sculpting).  
It sizes triangles as needed (usually smaller on the mesh boundary) resulting in a better quality result. 

![min angle](https://github.com/Pullusb/images_repo/raw/master/tess_minimum_angle_2fps.gif)


External contour only : Discard holes or internal shapes, take only external contour of shapes
![holes](https://github.com/Pullusb/images_repo/raw/master/tess_hole_shape.png)


Gift wrap : Close the convex hull of the shape  

![gift wrapping](https://github.com/Pullusb/images_repo/raw/master/tess_convex_hull.png)

UV mask : Generate geometry only on parts enclosed in the UV quad. If False the whole texture is meshed in 3D space  

![uv limits](https://github.com/Pullusb/images_repo/raw/master/tess_uv_masking.png)

True Delaunay algo : Use another algorithm for delaunay triangulation (conforming delaunay instead of constrained delaunay Triangle module settings)  
Not so different, this algorythm make sure voronoï cells are at center of each triangle (can force generation of super tiny tris to comply to this rule)  

<!-- discarded :  incremental algo : Use incremental algorithm instead of divide-and-conquer (Triangle module settings)   -->

## Credits

The addon is heavily inspired by the [tesselate addon within the rigging scripts released by "les fée spéciales" studio](https://github.com/LesFeesSpeciales/blender-rigging-scripts).  

Here are the main upgrade:
- works in 2.8
- use Opencv module instead of Scikit/Skyimage (still use Triangle for tesselation)
- better handling of holes when there are nested shapes (might still not be perfect though).
- expose a lot of options for tweaking (This also makes it a lot less stable...).
- Support (slighly) deformed UVs

## Installation of needed module  

This addons use two python modules to work. You have to install _triangulate_ and _opencv_.  
Be sure to download a version compatible with Blender 2.80's python 3.7.
This means you ought to have the same python version installed and use _pip install_ through this version. If you don't know how to do that look at "how to install module for specific python version".  

If you install through pip with `pip install opencv-python` and `pip install triangle`.  
You should see the folders of the modules in the python installation directory.  
In `<python_version>/lib/sites-packages`.  
Now copy opencv and triangle to the Blender module folder. If you don't know we're your Blender modules folders are, look at. "Were are the scripts folder in Blender". (Or video "addons and script installation in Blender", version quick or long).  

Opencv should have install some library dependency that you can see listed in console at pip install (make sure to copy them as well)

About module installation :  
If _triangle_ doesn't install properly with pip, follow [this tutorial on module installation by Damien Picard](http://lacuisine.tech/blog/2017/10/19/how-to-install-python-libs-in-blender-part-1/).


### Where ?
Panel in sidebar : 3D view > sidebar 'N' > Tool > Tex plane tesselation


### Note

If there is any uv_project modifier in stack, those will be applied automatically before running _(maybe add a checkbox to choose to bypass...)_

---

## Todo:
- demo : make a demo

- stability : problem... sometimes crash blender:  
   - exception happen inside Triangle module when PSLG polygon is pased with certains settings or images, dont really know how to fix it...

- puppet mode : make a planar snap bone creation tool to make a quick AE puppet style mode of rigging

- add a pixel margin on image np.array (usefull when no alpha between borders). can be an option True by default
   - This exists if the tesselate addon by les fées spéciale.

- check if mesh faces are co-planar
#https://blender.stackexchange.com/questions/107357/how-to-find-if-geometry-linked-to-an-edge-is-coplanar


Optional todo:

- for options: pix_margin and pix_clean, change pixel values to a value relative to the shortest side of the source image (percentage ?)

- support even more messed-up UVs (almost OK but homography with perspective not always giving accurate results with big deformation)

- Test if vertex of the mesh are [coplanar](https://blender.stackexchange.com/questions/107357/how-to-find-if-geometry-linked-to-an-edge-is-coplanar) avec avertissement si ce n'est pas le cas.

- maybe put a default less dense mesh

- choose the texture source if multiple in shader ? (or ensure this is the one connected to surface)

- add presets ?


<!-- ### Change log:
  2020/02/09 (1, 0, 2):
  - Main version ready -->