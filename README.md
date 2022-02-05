# Tesselate texture plane
Blender addon - Tesselate texture plane  
  
**[Download latest](https://github.com/Pullusb/Tesselate_texture_plane/archive/master.zip)**
  
### [Demo Youtube](https://youtu.be/bCw7BN5J8Pk)


### /!\ IMPORTANT NOTES:

  - **Running blender as admin** might be needed at first activation to enable auto-installation of needed third party modules (tested on windows)
  - **save before use !** Triangulating can crash with some images and/or settings (stable in "contour only" mode).

If it crash on your image, retry with different settings.  

This addon need [opencv](https://pypi.org/project/opencv-python/) and [triangle](https://rufat.be/triangle/) modules
At first activation, it will try to make an automatic download/installation of modules and their dependancies, if `modules` folder associated to your installation.  
If this doesn't work, [go below to manual install](#-manual-installation-of-needed-module).  

### Want to support my work ? check those links

- my [gumroad](https://pullusb.gumroad.com) store
- my [blender market](https://blendermarket.com/creators/pullup) page
- Other mean to support [here](http://www.samuelbernou.fr/donate)

---  

## Description  

Automatically tesselate a opaque zone of a texture plane mesh.  
You can also just create the contour without tesselation.  
The main purpose of the tesselation is to have a some vertices density in the mesh to be able to do "puppet" deformations with some rigging (similar to the puppet tool of After Effects).

Example:  
With _import image as plane_ addon, I imported this Ryu drawing image (found randomly on the web for test purpose).
"contour only" option generate a mesh with stripped alpha,  
otherwise it generate the same contour but fill it with internal triangles.

![demo](https://github.com/Pullusb/images_repo/raw/master/tess_usage_exemple.png)

And after some rig/skinning/animation...  

![paper ryu rig](https://github.com/Pullusb/images_repo/raw/master/paper_ryu-maillage_rig.png)

...I got _Paper ryu_!  

![paper ryu idle](https://github.com/Pullusb/images_repo/raw/master/paper_ryu-idle_20fps.gif)

_"Papeeer tatsumaki!"_  


![paper ryu idle](https://github.com/Pullusb/images_repo/raw/master/paper_ryu-tatsumaki_20fps.gif)

Here is the anim pose with wire

![paper ryu rig idle anim](https://github.com/Pullusb/images_repo/raw/master/paper_ryu-idleGl_20fps.gif)



> Ryu drawing by [steamboy](https://www.deviantart.com/steamboy33) ([deviant art page](https://www.deviantart.com/steamboy33/art/Ryu-THird-Strike-HD-291153215)), he made a superb animation with pretty much the same technique in after effect (combining multiple layers), [see details here](https://mugenguild.com/forum/index.php?topic=139727.0)

---

### Options detail

**Contour only** : No tesselation, just mesh the contour of the shape and fill with one Ngon face per separated "island" (stable, use only open-cv module).  
Note : does not handle holes.  

![contour only](https://github.com/Pullusb/images_repo/raw/master/tess_contour_only.png)


Simplify contour : 0=disabled, Approximation treshold of the contour polygon.  
Higher value decimate the mesh contour shape.  
Note: More simplification need also more pixel margin to avoid cutting though image.  

![simplify](https://github.com/Pullusb/images_repo/raw/master/tess_simplify_contour_2fps.gif)


**Aeration** : Limit maximum tri area, low value means more density (more triangles).  

![aeration](https://github.com/Pullusb/images_repo/raw/master/tess_aeration_setting_2fps.gif)

**Pixel margin** : 0=disabled, Dilate contour around shapes.  

![pix margin](https://github.com/Pullusb/images_repo/raw/master/tess_pixel_margin_2fps.gif)

**Pixel cleaning** : Delete surface with pixel dimensions smaller than given pixel size. (Do not use if your source is pixel art).  
Usefull for cleaning unwanted rogues pixels, pushing too much can also erode surface angles.  
In the example black part have alpha at 0 (smart test image isn't it  ?)  

![pix clean](https://github.com/Pullusb/images_repo/raw/master/tess_pixel_cleaning_2fps.gif)


**Minimum angle** : 0=disabled, "quality mesh generation" value, augment the mimimum angle limitation.  
The value is not given as an angle but as min/max available. Basically means more triangle with better repartition.  
Note : This have a big influence on aeration ! Add more density variation (kind of like Dyntopo for sculpting).  
It sizes triangles as needed (usually smaller on the mesh boundary) resulting in a better quality result. 

![min angle](https://github.com/Pullusb/images_repo/raw/master/tess_minimum_angle_2fps.gif)


**External contour only** : Discard holes or internal shapes, detect only external contour of shapes.  

![holes](https://github.com/Pullusb/images_repo/raw/master/tess_hole_shape.png)


**Gift wrap** : Close the convex hull of the shape  

![gift wrapping](https://github.com/Pullusb/images_repo/raw/master/tess_convex_hull.png)

**UV mask** : Generate geometry only on parts enclosed in the UV quad. If False the whole texture is meshed in 3D space.  

![uv limits](https://github.com/Pullusb/images_repo/raw/master/tess_uv_masking.png)

**True Delaunay algo** : Use another algorithm for delaunay triangulation, conforming delaunay instead of constrained delaunay Triangle module settings. Results isn't so different, but this algorithm make sure voronoï cells are at center of each triangles (can force generation of super tiny tris by complying to this rule).  
<!-- discarded :  incremental algo : Use incremental algorithm instead of divide-and-conquer (Triangle module settings)   -->

### Additional notes

- If there is any uv_project modifier in stack, those will be applied automatically before running
- Any subdivision modifiers will be deleted before applying.

### Where ?

Panel in sidebar : 3D view > sidebar 'N' > Tool > Tex plane tesselation

## Credits

The addon is heavily inspired by the original [Tesselate addon](https://github.com/LesFeesSpeciales/blender-rigging-scripts) within the rigging scripts released by "les fée spéciales" studio.  

Here are the main differences:

- Compatible with Blender 2.8+/3.0+
- use Opencv module instead of Scikit+Skyimage (still use Triangle for tesselation)
- better handling of holes when there are nested shapes (might still not be perfect though).
- expose a lot of options for tweaking (This also makes it a lot less stable...).
- Support (slightly) deformed UVs

## Manual installation of needed module  

This addons use two python modules to work. You have to install _triangulate_ and _opencv_.  
Be sure to download a version compatible with Blender 2.80's python 3.7.
This means you ought to have the same python version installed and use _pip install_ through this version. If you don't know how to do that look at "how to install module for specific python version".  

If you install through pip with `pip install opencv-python` and `pip install triangle`.  
You should see the folders of the modules in the python installation directory.  
In `<python_version>/lib/sites-packages`.  
Now copy opencv and triangle to the Blender module folder. 
If you don't know we're your Blender modules folders are, look at "Were are the scripts folder in Blender". (Or video "addons and script installation in Blender", version quick or long).  

Opencv should have install some library dependency that you can see listed in console at pip install (make sure to copy them as well)

About module installation :  
If _triangle_ doesn't install properly with pip, follow [this tutorial on module installation by Damien Picard](http://lacuisine.tech/blog/2017/10/19/how-to-install-python-libs-in-blender-part-1/).

---

## Todo:

- stability : Sometimes crash blender:  
  - exception happen inside Triangle module when PSLG polygon is passed with certains settings or images, dont know how to fix it...

<!-- - add a pixel margin on image np.array (usefull when no alpha between borders). can be an option True by default
   - This exists if the tesselate addon by les fées spéciale.

- check if mesh faces are co-planar
#https://blender.stackexchange.com/questions/107357/how-to-find-if-geometry-linked-to-an-edge-is-coplanar


Optional todo:

- for options: pix_margin and pix_clean, change pixel values to a value relative to the shortest side of the source image (percentage ?)

- support even more messed-up UVs (almost OK but homography with perspective not always giving accurate results with big deformation)

- Test if vertex of the mesh are [coplanar](https://blender.stackexchange.com/questions/107357/how-to-find-if-geometry-linked-to-an-edge-is-coplanar) avec avertissement si ce n'est pas le cas.

- maybe put a default less dense mesh

- choose the texture source if multiple in shader ? (or ensure this is the one connected to surface)

- add presets ? -->


### Changelog

2.0.0

- compatible with blender 2.93 / 3.0.0 +
- improved module auto-install

1.1.0

- modules auto-install using pip (tested only on windows)
- UI: redo panel is now the same as sidebar UI  
- Exposed disply wireframe in the UI
- Added tracker URL infos

1.0.2

- Main version ready:
  - cv2 contour stable
  - some images/settings can still crash blender in tesselation (using triangle module)