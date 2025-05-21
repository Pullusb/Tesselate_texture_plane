# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
"name": "Tesselate texture plane",
"description": "Triangulate or cut selected textured mesh planes on opaque areas",
"author": "Samuel Bernou",
"version": (3, 1, 0),
"blender": (4, 0, 0),
"location": "3D view > right toolbar > Tesselate tex plane",
"warning": "Tesselate mode can crash (Save before use). 'Contour only' mode stable",
"wiki_url": "https://github.com/Pullusb/Tesselate_texture_plane",
"tracker_url": "https://github.com/Pullusb/Tesselate_texture_plane/issues",
"category": "3D View"
}


import bpy
import numpy as np
import bmesh
from mathutils import Vector, Matrix
from time import time
from pathlib import Path

DEPENDENCIES = {
    ('cv2', 'opencv-python'), # opencv-contrib-python
    ('triangle', 'triangle'),
}

def module_can_be_imported(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False

# Function to check which dependencies are missing
def get_missing_dependencies():
    return [module for module, _ in DEPENDENCIES if not module_can_be_imported(module)]

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import triangle # triangle doc >> https://rufat.be/triangle/API.html
except ImportError:
    triangle = None

## addon basic import shortcuts for class types and props
from bpy.props import (IntProperty,
                        StringProperty,
                        BoolProperty,
                        FloatProperty,
                        EnumProperty,
                        CollectionProperty,
                        PointerProperty,
                        IntVectorProperty,
                        BoolVectorProperty,
                        FloatVectorProperty,
                        RemoveProperty,
                        )

from bpy.types import (Operator,
                       Panel,
                       PropertyGroup,
                       UIList,
                       AddonPreferences,
                       )

def debug_display_img(img, img_name, width, height, base_one=True):
    blimg = bpy.data.images.get(img_name)
    if blimg:
        #print('x', img.shape[0], blimg.size[1], ' y', img.shape[1], blimg.size[0])
        if img.shape[0] == blimg.size[1] and img.shape[1] == blimg.size[0]:
            pass
        else:
            bpy.data.images.remove(blimg)
            blimg = bpy.data.images.new(img_name, width=width, height=height, alpha=True)#, float_buffer=False
    else:
        blimg = bpy.data.images.new(img_name, width=width, height=height, alpha=True)#, float_buffer=False

    start = time()
    if not base_one:#if image is base 255
        img = img / 255
    ravelled = img.ravel()
    blimg.pixels = ravelled #Loooooooong !
    print(f'pixel2img time : {time() - start} secs')

def back_to_tex_node(n):
    for s in n.inputs:
        for link in s.links:
            if link.from_node.type == 'TEX_IMAGE':
                return link.from_node
            else:
                return back_to_tex_node(link.from_node)

#TODO : if use in an operator, make a popup with the lists of textures (or find a way to filter whats "surface")
def get_tex(o):
    '''
    Get texture use in the surface shader.
    Return first texture found by going recursively going back from shader output node
    '''

    tex_node = None
    if o.active_material:
        if o.active_material.node_tree:
            nodes = o.active_material.node_tree.nodes

            out_nodes = [i for i in nodes if i.type == 'OUTPUT_MATERIAL' and i.inputs['Surface'].is_linked]
            if len(out_nodes) > 1:
                print('multiple output node found !')
            if out_nodes:
                out = out_nodes[0]
                tex_node = back_to_tex_node(out)
                if tex_node:
                    #print(tex_node)
                    print('texture found to work on :', tex_node.image)
    return tex_node


# Draw a point
def draw_point(img, p, color ) :
    '''
    img to paint (np array)
    point coordinate
    radius of the circle
    color tuple
    thickness: negative value means fill (cv2.FILLED = -1)
    linetype: 4, 8, 16 (cv2.LINE_AA = 16, anti-aliased) 4 and 8 are aliased
    '''
    #print('point:', p)
    cv2.circle( img, p, 1, color, cv2.FILLED, cv2.LINE_AA, 0 )

def draw_tuple_line(img, a, b, color, thickness=1):
    ''' image to draw on
        start coord
        end coord
        color tuple,
    '''
    cv2.line(img, tuple((int(a[0]), int(a[1]))), tuple((int(b[0]), int(b[1]))), color, thickness, cv2.LINE_AA, 0)

def uv_from_vert_first(uv_layer, v):
    '''https://blender.stackexchange.com/questions/49341/how-to-get-the-uv-corresponding-to-a-vertex-via-the-python-api'''
    for l in v.link_loops:
        uv_data = l[uv_layer]
        return uv_data.uv
    return None

def get_corners(ob):
    '''
    Get corner of the current planar/trapezoidal shape in UV (4 point at least)
    In the right order (starting bottom left and going Counter-Clockwise)
    return (coord list, index list)
    '''
    
    me = ob.data
    
    if len(me.vertices) == 4:#if there is only 4 points
        # This condition is not necessary but "maybe" faster for finding corresponding vert id and uv id when 4 verts only
        
        points = [p.uv for p in ob.data.uv_layers.active.data]
        #find the four corner (only works with a 4 point shape)
        bottom, up = sorted(points, key=lambda x: x[1])[:2], sorted(points, key=lambda x: x[1])[2:]
        bl,br = sorted(bottom, key=lambda x: x[0])
        ul,ur = sorted(up, key=lambda x: x[0])
        sorted_points = [bl,br,ur,ul]
        points_index = [points.index(i) for i in sorted_points]#get uv index from uv points
        return sorted_points, points_index

    ## more than 4 points, use bmesh to get corner points if plane is subdivided.
    bm = bmesh.new()#bm = bmesh.from_edit_mesh(me)#for edit mode
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active

    ## Get only points that have only 2 connecting edges (corner)
    ## Need to copy (or recast to vector) otherwise value is a pointer and change. 
    ## Generate a list with sublist containing vector coordinate and associated mesh index : [ [Vector(1,1), 3] , ...]
    points = [[uv_from_vert_first(uv_layer, v).copy(), i] for i, v in enumerate(bm.verts) if len(v.link_edges) == 2]

    #clean bmesh
    bm.free()
    del bm

    if len(points) != 4: print('!!!', ob.name, ' -> Not 4 points found for UV corner :', len(points))
    
    ## filter corner according to coordinate
    ## X[0] -> the vector, so (x[0][0] = x, x[0][1] = y)
    bottom, up = sorted(points, key=lambda x: x[0][1])[:2], sorted(points, key=lambda x: x[0][1])[2:]
    ul,ur = sorted(up, key=lambda x: x[0][0])
    bl,br = sorted(bottom, key=lambda x: x[0][0])


    corner_list = [bl,br,ur,ul]
    sorted_points = [i[0] for i in corner_list]#coordinate list
    points_index = [i[1] for i in corner_list]#mesh index list


    ### Find corresponding UV index for given vert index
    ### https://docs.blender.org/api/blender_python_api_2_78_release/bpy.types.Mesh.html
    ### https://www.blender.org/forum/viewtopic.php?t=25607

    uv_layer = me.uv_layers.active.data

    ## method 1 (ugly), check if vector is the same 
    """ id_list = []
    for p in sorted_points:
        for i, uvloop in enumerate(ob.data.uv_layers.active.data):
            print('uvloop: ', uvloop)
            print(dir(uvloop))
            if uvloop.uv == p:
                id_list.append(i)
                break
    print('id_list: ', id_list) """

    ## method 2 (good) 
    uv_points_index = []
    for pi in points_index:
        for poly in me.polygons:
            for loop_index in poly.loop_indices:
                if me.loops[loop_index].vertex_index == pi:
                    print('id vert/uv: %s -> %s' % (pi, loop_index))
                    uv_points_index.append(loop_index)

    #uv_points_index = [0,1,2,3]#usually the right order for a 4 point basic plane
    return (sorted_points, uv_points_index)


def get_plane_matrix(ob):
    """Get object's polygon local matrix from uvs"""

    '''
    #find the four corner (only works with a 4 point shape)
    points = [p.uv for p in obj.data.uv_layers.active.data]
    bottom, up = sorted(points, key=lambda x: x[1])[:2], sorted(points, key=lambda x: x[1])[2:]
    ul,ur = sorted(up, key=lambda x: x[0])
    bl,br = sorted(bottom, key=lambda x: x[0])
    p0 = points.index(bl)
    px = points.index(br)
    py = points.index(ul)
    '''

    points, points_index = get_corners(ob)
    #bl,br,ur,ul = points[:]#unpack corners

    p0 = points_index[0]#bl
    px = points_index[1]#br
    py = points_index[3]#ul

    p0 = ob.data.vertices[ob.data.loops[p0].vertex_index].co
    px = ob.data.vertices[ob.data.loops[px].vertex_index].co - p0
    py = ob.data.vertices[ob.data.loops[py].vertex_index].co - p0

    rot_mat = Matrix((px, py, px.cross(py))).transposed().to_4x4()
    trans_mat = Matrix.Translation(p0)
    mat = trans_mat @ rot_mat

    return mat


def generate_mesh(obj, res, npimg):
    '''pass original object (obj) and triangulate dic (res)'''
    mesh = obj.data

    mat = get_plane_matrix(obj)#obj.matrix_world

    shape = npimg.shape

    #######
    #### / check homography deformation geometry
    ###### 

    #https://stackoverflow.com/questions/55055655/how-to-use-cv2-perspectivetransform-to-apply-homography-on-a-set-of-points-in
    hg = None
    sorted_points, points_index = get_corners(obj)

    ## pts_src and pts_dst are numpy arrays of points in source and destination images. need at least 4 corresponding points. 
    dest = np.array([[0,0], [1,0], [1,1], [0,1]])
    hg, status = cv2.findHomography( np.array(sorted_points), dest, cv2.RANSAC)#last argument is optional
    #print('cv2.findHomography( np.array(sorted_points), dest, cv2.RANSAC): \n', cv2.findHomography( np.array(sorted_points), dest, cv2.RANSAC))
    # hg = cv2.getPerspectiveTransform(np.array(sorted_points), dest)
    
    ### generate bmesh

    bm = bmesh.new()

    vert_list = []
    for v_co in res['vertices']:
        # x = v_co[1] * shape[0] / shape[1]#old-> needed np.rot on pixel shape
        # y = (1-v_co[0])#old-> needed np.rot on pixel shape
        x = (v_co[0])
        y = v_co[1] * shape[1] / shape[0]

        pt = [x,y]

        vert_list.append(pt)
        # bm.verts.new([pt.x, pt.y, 0])
    
    # Create vertices:
    if vert_list:
        for pt in vert_list:
            bm.verts.new([pt[0], pt[1], 0])
    else:
        print('no vertice to generate, (in mesh generation)')

    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    for s in res['segments']:
        bm.edges.new((bm.verts[s[0]], bm.verts[s[1]]))

    for face in res['triangles']:
        bm.faces.new([bm.verts[i] for i in face])

    edges = []
    for edge in bm.edges:
        edge.select = True
        edges.append(edge)

    bmesh.ops.remove_doubles(bm, verts=bm.verts)

    bm.verts.index_update()

    # UVs
    uv_layer = bm.loops.layers.uv.verify()
    # uv_layer_name = uv_layer.name#unused
    # bm.faces.layers.tex.verify()#tex not exists !
    for f in bm.faces:
        for l in f.loops:
            luv = l[uv_layer]
            luv.uv = l.vert.co.xy

    print()
    # Transform 3D mesh verts coordinates
    for v in bm.verts:
        # test : move by 2 in X - v.co = mat @ Vector((v.co[0] + 2, v.co[1], 0)) 

        if hg is not None:
            pts = np.float32([v.co[0], v.co[1]]).reshape(-1,1,2)#method 1 to reshape
            #pts = np.array([v.co], np.float32)#method 2 to reshape

            newp = cv2.perspectiveTransform(pts, hg)#apply homography matrix to the point.
            npt = newp.tolist()[0][0]
            v.co = mat @ Vector((npt[0],npt[1], 0))

        else:
            v.co = mat @ v.co

    bm.to_mesh(mesh)
    mesh.update()


def tesselate(obj, contour_only=False, simplify=0.0010, pix_margin=2, min_angles=20, aeration=0.005, external_only=False, pix_clean=0, true_delaunay=False, algo_inc=False, gift_wrap=False, uv_mask=True):
    """
    Take a textured planar objects, triangulate the mesh based on texture aplha and replace it

    Args:
    - simplify : simplification (cv:approximate) 
    0=disabled 0.0010 ~ 0.0020 (approximation treshold)
    
    - pix_margin : dilate margin, pixel margin (cv:dilate)
    0=disabled  default=2 (pixel value)

    - min_angle : no angles smaller value given (higher value allow more triangles)
    0=disable default=20 10~30 (degree) (real max is 35)

    - aeration/tri_max_size/sparse : limit maximum tri area, low value means more density, add more triangles 
    0=disable default=0.005 :  0.0001(super dense)~0.01(low density) (0.001 seem like a mid value) (area size value) 
    seems more crashy at 0.001 and below... (might be too dense)

    - only_external : dont fill hole or internal shapes only external contour of shapes
    default=False (Bool)


    Optional Args
    - pix_clean/clean_dust : delete surface with pixel dimensions smaller than given size (cv:morph -> erode then re-dilate)
    default=0 (pixel value)

    - true_delaunay :  conforming delaunay (instead of constrained delaunay)
    default=False (Bool)

    - algo_inc : Use incremental algotrythm instead of divide and conquer

    Super optional Args
    - gift_wrap/convex_hull : close the convex hull of the shape
    default=False (Bool)

    - uv_mask : Mesh only what is enclosed in UV quad. if False the whole texture is meshed
    default=True (Bool)
    """
    
    t_convert = time()#Dbgt
    
    #Get texture associated with current object
    tex = get_tex(obj)
    if not tex:
        print('no texture found in nodes')
        return
    b_img = tex.image

    #b_img.pixels is a flat array of pixels, #reshape it to correct np array model (x,y,4 channel) 
    orig_img = np.array(b_img.pixels).reshape([b_img.size[1], b_img.size[0], 4])

    img = orig_img.copy()#make a copy to work on (check alpha).
    
    #img = np.flipud(img) # flip along y axis #corrected, no need anymore
    ### if need to print full array in console >> with np.printoptions(threshold=np.inf): print('img dtype, shape:', img.dtype, img.shape)

    alphaimg = img[..., 3]#keep only alpha dimension

    thresh_mask = alphaimg[...] < 0.01#get True / False array by evaluating alpha (True where alpha is 0)

    alphaimg[:] = 1.0#make an array with same size filled with 1

    alphaimg[thresh_mask] = 0.0#array mask : this apply 0 where a True collide a '1', and a 1 where a False collide a '1'

    alphaimg = np.array(alphaimg, dtype=np.uint8)#FindContours need a int-8 type array (so only 0,1 int values)

    #alphaimg = np.rot90(alphaimg, k=-1)#old-rotate Clockwise... (Because vertex position was messed up in the end, no need)
    print('to binary image: {:.3f}s'.format(time() - t_convert))#Dgbt
    
    #######
    ### ---UV mask (otherwise the whole image get meshed since detection happen on the whole image)
    ######  

    if uv_mask:
        t_uvmask = time()#Dbgt
        uvzone = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8 )

        uvpoly, uvpoly_index = get_corners(obj)
        # uvpoly = [p.uv for p in obj.data.uv_layers.active.data][:4]

        uvpoly.append(uvpoly[0])#close shape by adding first point at last

        #convert uv coord (0.0 to 1.0) to int pixel coord of image in a numpy array
        uvpolypixel = np.array([[int(p[0]*img.shape[1]), int(p[1]*img.shape[0])] for p in uvpoly])

        
        """ # debug draw poly found on image in a 'uv_zone' texture 
        draw_tuple_line(img, uvpolypixel[0], uvpolypixel[1], (0,255,0,1), thickness=2)
        draw_tuple_line(img, uvpolypixel[1], uvpolypixel[2], (0,255,0,1), thickness=2)
        draw_tuple_line(img, uvpolypixel[2], uvpolypixel[3], (0,255,0,1), thickness=2)
        draw_tuple_line(img, uvpolypixel[3], uvpolypixel[4], (0,255,0,1), thickness=2)
        debug_display_img(img, 'uv_zone', img.shape[1], img.shape[0], base_one=True) """
        

        #paint uv zone in pixel image
        ## args : img, list of contour, -1: all contour (shoulg be only one in this case), 1:"paint" value '1', -1: fill contour
        cv2.drawContours(uvzone, [uvpolypixel], -1, 1, -1) 
        alphaimg *= uvzone#mask zone

        print('uv_maskink: {:.3f}s'.format(time() - t_uvmask))#Dgbt


    #######
    ### --- Alterate pixel shape here before converting to poly---
    ######

    if pix_clean:
        t_pixclean = time()#Dbgt
        #delete supertiny isolated zones
        kernel = np.ones((pix_clean, pix_clean), np.uint8)
        alphaimg = cv2.morphologyEx(alphaimg, cv2.MORPH_OPEN, kernel)
        print('pixclean: {:.3f}s'.format(time() - t_pixclean))#Dgbt

    if pix_margin:
        t_pixmargin = time()#Dbgt
        #pixel dilatation/erode
        kernel = np.ones( ( abs(pix_margin), abs(pix_margin) ), np.uint8)
        if pix_margin > 0:
            alphaimg = cv2.dilate(alphaimg, kernel, iterations = 1)
        else:#if negative, erode
            alphaimg = cv2.erode( alphaimg, kernel, iterations = 1)
        print('pixmargin: {:.3f}s'.format(time() - t_pixmargin))#Dgbt

    # debug_display_img(alphaimg, 'tesselation', alphaimg.shape[0], alphaimg.shape[1], base_one=False)

    #######
    ### --- Contour handling with openCV
    #######

    if external_only:
        contour_mode = cv2.RETR_EXTERNAL # only full shapes (no holes)
    else:
        #contour_mode = cv2.RETR_TREE # full hiearchy (No need)
        contour_mode = cv2.RETR_CCOMP # 2 level hierachy (shape and holes)

    t_getcontour = time()#Dbgt
    contours, hierarchy = cv2.findContours(alphaimg, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
    print('getcontour: {:.3f}s'.format(time() - t_getcontour))#Dgbt


    # debug : draw contour in opencv in image editor
    # cv2.drawContours(img, contours, -1, (255,255,255,1), 2)
    # debug_display_img(img, 'contour', img.shape[0], img.shape[1], base_one=True)

    cnt_dic = {'vertices':  [],
            'segments':  [],
            'holes':     [],
            'triangles': [],
            }

    print('detected contours:',  len(contours))
    if not len(contours):
        print('NO CONTOURS DETECTED, aborting for obj', obj.name)
        return
    
    prev_cnt_index = 0

    t_approxshape = time()
    for c, h in zip(contours, hierarchy[0]): # hierachy is a nested list
        cnt = c[:,0] # remove the upper level in nested list for coords in array [[x, y]] -> [x,y]
        np.append(cnt, cnt[0]) # close shape

        is_hole = h[3] != -1# if it has a parent index then it's a hole (with RETR_CCOMP)
        # print('cnt : dtype', cnt.dtype, 'shape:', cnt.shape)#Dbg

        ### contour simplification, perform approximation per contour
        #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-approximation
        if simplify:
            # print('len raw  cnt: ', len(cnt))#before
            cnt = cv2.approxPolyDP(cnt, simplify*cv2.arcLength(cnt,True), True)[:,0]
            # print('len aprx cnt: ', len(cnt))#after
        

        # Divide to get a 1x1 square. map between 0 and 1
        cnt = cnt.astype(float) # convert polygons pixels coordinate to float so it can be mapped between 0,1
        for axis in range(2):
            cnt[..., axis] /= float(alphaimg.shape[axis])
    
        cnt[..., 0] *= alphaimg.shape[0] / alphaimg.shape[1] # Get aspect ratio back
    
 
        ### add vertices to contour dic.
        cnt_dic['vertices'].extend(cnt)# extend to add every new contour as a flat list
        
        if contour_only:#face are directly using vertice order
            cnt_dic['triangles'].append([prev_cnt_index + i for i,c in enumerate(cnt)])

        # create segment list (index of the two vertices in dic['vertices'])
        seglist = 'segments'
        for pt_i in range(len(cnt)-1):
            cnt_dic[seglist].append(
                [pt_i + prev_cnt_index, pt_i + prev_cnt_index + 1] )
        
        #add last segment
        cnt_dic[seglist].append([len(cnt) - 1 + prev_cnt_index, prev_cnt_index])

        prev_cnt_index += len(cnt)# next contour will continue after last vertex index

        if is_hole:
            # to add a hole, specify a point inside the hole.
            # get x-y pair with the lowest y and create a point 1 pixel above (dirty but works in most cases...) #CBB
            lowest_y_pair = min(cnt, key=lambda x: x[1])
            cnt_dic['holes'].append( [lowest_y_pair[0], lowest_y_pair[1] + 0.0001] )

    print('approxshape: {:.3f}s'.format(time() - t_approxshape))#Dgbt

    ## kill unfilled keys in dic
    for k in ('holes', 'triangles', 'segments'):
        if not cnt_dic[k]:
            del cnt_dic[k]

    ### If contour only, no triangulation - direct mesh and return
    if contour_only:
        t_meshing = time()#Dbgt
        generate_mesh(obj, cnt_dic, orig_img)
        print('meshing contour: {:.3f}s'.format(time() - t_meshing))#Dgbt
        print('Done')
        return

    #######
    ### --- Tesselation with Triangle
    #######

    ''' Tri opts details :
        p - Triangulates a Planar Straight Line Graph.
        r - Refines a previously generated mesh.
        q - Quality mesh generation with no angles smaller than 20 degrees. An alternate minimum angle may be specified after the q.
        a - Imposes a maximum triangle area constraint. A fixed area constraint (that applies to every triangle) may be specified after the a, or varying areas may be read from the input dictionary.
        c - Encloses the convex hull with segments.
        D - Conforming Delaunay: use this switch if you want all triangles in the mesh to be Delaunay, and not just constrained Delaunay; or if you want to ensure that all Voronoi vertices lie within the triangulation.
        X - Suppresses exact arithmetic.
        S - Specifies the maximum number of added Steiner points.
        i - Uses the incremental algorithm for Delaunay triangulation, rather than the divide-and-conquer algorithm.
        F - Uses Steven Fortune’s sweepline algorithm for Delaunay triangulation, rather than the divide-and-conquer algorithm.
        l - Uses only vertical cuts in the divide-and-conquer algorithm. By default, Triangle uses alternating vertical and horizontal cuts, which usually improve the speed except with vertex sets that are small or short and wide. This switch is primarily of theoretical interest.
        s - Specifies that segments should be forced into the triangulation by recursively splitting them at their midpoints, rather than by generating a constrained Delaunay triangulation. Segment splitting is true to Ruppert’s original algorithm, but can create needlessly small triangles. This switch is primarily of theoretical interest.
        C - Check the consistency of the final mesh. Uses exact arithmetic for checking, even if the -X switch is used. Useful if you suspect Triangle is buggy.
    '''

    tri_opts = 'p'#base, PSLG mode

    if algo_inc:
        tri_opts += 'i'

    ## Tests for stability issue
    # tri_opts += 'C'# with Consistency check (exact arythmetics)... TEST, does not prevent crashs to happen
    # tri_opts += 'X'# Suppresses exact arithmetic ('C' takes precedence) TEST, seems even more crashy

    # -> https://www.cs.cmu.edu/~quake/tripaper/triangle2.html


    if min_angles:
        tri_opts += 'q' + str(min_angles)
    # else:## stability TEST, with or without quality mesh
    #     tri_opts += 'q'#just add the q ?

    if aeration:
        tri_opts += 'a' + "{:.4f}".format(aeration)# str(aeration)
    
    if true_delaunay:
        tri_opts += 'D'
    
    if gift_wrap:
        tri_opts += 'c'


    t_triangulate = time()
    print('triangulate with opts:', tri_opts)
    
    # try:# Crash with access violation, try block useless
    res = triangle.triangulate(cnt_dic, opts=tri_opts) #opts='piqa0.0005') #'segments':cnt (force this segment to be use)
    # except Exception as e:
    #     print('triangulation has failed on object')
    #     return

    print('triangulate: {:.3f}s'.format(time() - t_triangulate))#Dgbt

    print('points numbers:', len(res['vertices']))

    if not 'segments' in res.keys():
        print('no segments !'.upper())

    if not 'triangles' in res.keys():
        print('no triangles !'.upper())


    # draw image pixel version of resulted triangulation for debug purpose
    """ 
    for t in res['triangles']:
        draw_tuple_line(img, res['vertices'][t[0]], res['vertices'][t[1]], (0,255,0,1))#green
        draw_tuple_line(img, res['vertices'][t[1]], res['vertices'][t[2]], (0,255,0,1))
        draw_tuple_line(img, res['vertices'][t[2]], res['vertices'][t[0]], (0,255,0,1))


    for p in res['vertices']:#.tolist()
        draw_point(img, tuple( ( int(p[0]), int(p[1]) ) ), (0,0,255, 1))#Blue

    for s in res['segments']:
        draw_tuple_line(img, res['vertices'][s[0]], res['vertices'][s[1]], (255,0,0,1), thickness=1)#red

    ## debug display tesselation as image in image editor 
    #debug_display_img(img, 'tesselation', img.shape[0], img.shape[1], base_one=True)#need to pass True if img on base float 0.0 to 1.0
    # debug_display_img(img, 'tesselation', img.shape[0], img.shape[1], base_one=False)
    """

    t_meshing = time()#Dbgt
    generate_mesh(obj, res, orig_img)
    print('meshing: {:.3f}s'.format(time() - t_meshing))#Dgbt

    print('Done')


def transfer_value(Value, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((Value - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

class TESS_OT_tesselate_plane(Operator):
    bl_idname = "mesh.tesselate_plane"
    bl_label = "Tesselate tex plane"
    bl_description = "Tesselate selected texture plane objects"
    bl_options = {"REGISTER", "UNDO"}#UNDO to change value, slight risk of crashing everything.. 

    @classmethod
    def poll(cls, context):
        return context.selected_objects

    contour_only : BoolProperty(name="Contour only", 
    description="No tesselation, just mesh contour of shapes (use only open-cv module)", 
    default=False)

    simplify : FloatProperty(name="Simplify contour", 
    description="0=disabled, Approximation treshold\nHigher value simplify contour shape\nNote: More simplify need also more pixel margin to avoid cutting though image", # (less external points, so less triangle density)
    default=1, min=0.0, max=100.0, step=1, precision=1, subtype='PERCENTAGE', unit='NONE')

    pix_margin : IntProperty(name="Pixel margin", 
    description="0=disabled, Dilate contour around shape", 
    default=2, min=0, max=800, soft_min=0, soft_max=50, step=1, subtype='PIXEL')

    # min_angles : IntProperty(name="Minimum angle", 
    # description="0=disabled, No angles smaller value given (higher value allow more triangles)", 
    # default=20, min=0, max=35, soft_min=0, soft_max=34, step=1, subtype='ANGLE')

    min_angles : IntProperty(name="Minimum angle", 
    description="0=disabled, Allow more triangle (augment the mimimum angle limitation)", 
    default=20, min=0, max=99, step=1)#100 run forever

    aeration : FloatProperty(name="Aeration", 
    description="Limit maximum tri area, low value means more density, add more triangles", 
    default=5, min=-50, max=300, soft_min=0.0, soft_max=100.0, step=3, precision=2, subtype='PERCENTAGE', unit='NONE')

    external_only : BoolProperty(name="External contour only", 
    description="Discard holes or internal shapes, take only external contour of shapes", 
    default=True)


    #--- optional args
    pix_clean : IntProperty(name="Pixel cleaning", 
    description="Delete surface with pixel dimensions smaller than given size. (Do not use if your source is pixel art)",
    default=2, min=0, max=250, soft_min=0, soft_max=50, step=1, subtype='PIXEL')

    true_delaunay : BoolProperty(name="True Delaunay algo", 
    description="Use another algorithm for delaunay triangulation (conforming delaunay instead of constrained delaunay Triangle module settings)", 
    default=False)

    algo_inc : BoolProperty(name="incremental algo", 
    description="Use incremental algorithm instead of divide-and-conquer (Triangle module settings)", 
    default=False)

    gift_wrap : BoolProperty(name="Gift wrap", 
    description="Close the convex hull of the shape", 
    default=False)

    uv_mask : BoolProperty(name="UV mask", 
    description="Generate geometry only on parts enclosed in the UV quad. If False the whole texture is meshed in 3D space", 
    default=True)


    def execute(self, context):
        ## Pre-flight check: Raise error if Triangle is missing and contour_only is not enabled
        triangle_missing = not module_can_be_imported('triangle')
        
        if triangle_missing and not self.contour_only:
            self.report({'ERROR'}, 
                       "Triangle module is required for tesselation mode. "
                       "Please enable 'Contour only' mode or install Triangle from addon preferences.")
            return {'CANCELLED'}

        #map/translate values from human readable to triangle and opencv value
        # Value = self.Value if self.Value == 0 else transfer_value(self.Value, OldMin, OldMax, NewMin, NewMax)
        
        simplify = self.simplify if self.simplify == 0 else transfer_value(self.simplify, 0.0, 100.0, 0.0001, 0.0025)#0.0010, 0.0020

        min_angles = self.min_angles if self.min_angles == 0 else transfer_value(self.min_angles, 0.0, 100.0, 10, 35)

        # self.aeration if self.aeration == 0 else (disabling aearation to 0 max aeration !)
        aeration = transfer_value(self.aeration, 0.0, 100.0, 0.00001, 0.01)#0.0001(super dense)~0.01(low density) (0.001 seem like a mid value) #old : 0.0001, 0.01

        #debug prints
        """ print()
        print("-- Using values:")
        print(' ', self.simplify, '>', simplify)
        print(' ', self.min_angles, '>', min_angles)
        print(' ', self.aeration, '>', aeration)
        print(' ', external_only)
        print(' ', pix_clean)
        print(' ', true_delaunay)
        print(' ', gift_wrap)
        print(' ', uv_mask) """

        print(f"""-- Using values:
        simplify      :{self.simplify} > {simplify}
        min_angles    :{self.min_angles} > {min_angles}
        aeration      :{self.aeration} > {aeration}
        external_only :{self.external_only}
        pix_clean     :{self.pix_clean}
        true_delaunay :{self.true_delaunay}
        gift_wrap     :{self.gift_wrap}
        uv_mask       :{self.uv_mask}""")

        ##########
        #### --- START HERE !!!
        ##########

        all_start = time()
        only_four_verts_plane = False

        #TODO: add check if mesh faces are co-planar ?

        #iterate only on mesh type objects which are planes...
        if only_four_verts_plane:
            #...that have only 4 vertices
            objlist = [o for o in bpy.context.selected_objects if o.type == 'MESH' and len(o.data.vertices) == 4]
        else: 
            #...that have 4 vertices or more
            objlist = [o for o in bpy.context.selected_objects if o.type == 'MESH' and len(o.data.vertices) >= 4]


        for obj in objlist:
            start = time()
            print('\nObject to tesselate: ', obj.name)

            # check modifiers
            for m in obj.modifiers:
                print(m)
                ## remove (or disable) subdivision surface modifier
                if m.type == 'SUBSURF':
                    print('disabling:', m.name)
                    obj.modifiers.remove(m)#remove
                    # m.show_viewport = False
                    # m.show_render = False
                    continue

                #set active and apply UV modifier
                if m.type == 'UV_PROJECT':
                    print('appllying:', m.name)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.modifier_apply(modifier=m.name)
                    continue


            ### TESSELATE
            tesselate(obj,
            contour_only = self.contour_only,
            simplify = simplify,
            pix_margin = self.pix_margin,
            min_angles = min_angles,
            aeration = aeration,
            external_only = self.external_only,
            pix_clean = self.pix_clean,
            true_delaunay = self.true_delaunay,
            algo_inc = self.algo_inc,
            gift_wrap = self.gift_wrap,
            uv_mask = self.uv_mask
            )

            ### Default settings
            # tesselate(obj)

            ### Without holes, bigger aeration and some margin and pixel island cleaning
            # tesselate(obj, external_only=True, pix_margin=2)

            ### More dense (uniform)
            # tesselate(obj, aeration=0.0002)

            ### More dense (with more on side)
            # tesselate(obj, aeration=0.0004, min_angles=32)

            ### Higher contour simplification, relatively dense mesh, erase pixels zones with size < 5px
            # tesselate(obj, simplify=0.0018, aeration=0.002, pix_clean=5)

            print('time: {:.3f}s'.format(time() - start))


        print('-All Done: {:.3f}s'.format(time() - all_start))
        #yeah !

        return {"FINISHED"}
    
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "contour_only")
        col.prop(self, "simplify")

        scol = col.column()
        scol.enabled = not self.contour_only
        scol.prop(self, "aeration")
        col = layout.column()

        col.separator()
        col.prop(self, "pix_margin")
        col.prop(self, "pix_clean")

        scol = col.column()
        scol.enabled = not self.contour_only
        scol.prop(self, "external_only")
        col = layout.column()

        col.prop(self, "uv_mask")

        scol = col.column()
        scol.label(text='Triangle extra:')
        scol.prop(self, "gift_wrap")
        scol.prop(self, "true_delaunay")
        scol.prop(self, "min_angles")
        # scol.prop(self, "algo_inc")

    def invoke(self, context, event):
        # panel to operator variable attribution (for the redo)
        self.contour_only = context.scene.ttp_props.contour_only
        self.simplify = context.scene.ttp_props.simplify
        self.pix_margin = context.scene.ttp_props.pix_margin
        self.min_angles = context.scene.ttp_props.min_angles
        self.aeration = context.scene.ttp_props.aeration
        self.external_only = context.scene.ttp_props.external_only
        self.pix_clean = context.scene.ttp_props.pix_clean
        self.true_delaunay = context.scene.ttp_props.true_delaunay
        self.gift_wrap = context.scene.ttp_props.gift_wrap
        self.uv_mask = context.scene.ttp_props.uv_mask
        # self.algo_inc = context.scene.ttp_props.algo_inc

        return self.execute(context)

class TESS_props_group(PropertyGroup):
    '''Duplicate props of the operator to display in panel'''

    contour_only : BoolProperty(name="Contour only", 
    description="No tesselation, just mesh contour of shapes (use only open cv module)", 
    default=False)

    simplify : FloatProperty(name="Simplify contour", 
    description="0=disabled, Approximation treshold\nHigher value simplify contour shape\nNote: More simplify need also more pixel margin to avoid cutting though image", # (less external points, can also trigger less triangle density)
    default=1, min=0.0, max=100.0, step=1, precision=1, subtype='PERCENTAGE', unit='NONE')

    pix_margin : IntProperty(name="Pixel margin", 
    description="0=disabled, Dilate contour around shape", 
    default=2, min=0, max=800, soft_min=0, soft_max=50, step=1, subtype='PIXEL')

    min_angles : IntProperty(name="Minimum angle", 
    description="0=disabled, Add more triangle by augmenting the mimimum angle limitation\nAdd more density variation", 
    default=20, min=0, max=99, step=1)#100 run forever

    aeration : FloatProperty(name="Aeration", 
    description="Limit maximum tri area, low value means more density, add more triangles", 
    default=5, min=-50, max=300, soft_min=0.0, soft_max=100.0, step=3, precision=2, subtype='PERCENTAGE', unit='NONE')

    external_only : BoolProperty(name="External contour only", 
    description="Discard holes or internal shapes, take only external contour of shapes", 
    default=True)

    #--- optional args
    pix_clean : IntProperty(name="Pixel cleaning", 
    description="Delete surface with pixel dimensions smaller than given size. (Do not use if your source is pixel art)",
    default=2, min=0, max=250, soft_min=0, soft_max=50, step=1, subtype='PIXEL')

    true_delaunay : BoolProperty(name="True Delaunay algo", 
    description="Use another algorithm for delaunay triangulation (conforming delaunay instead of constrained delaunay Triangle module settings)\nMake sure voronoï cells are at center of each triangle (can make little tris)", 
    default=False)

    algo_inc : BoolProperty(name="incremental algo", 
    description="Use incremental algorithm instead of divide-and-conquer (Triangle module settings, from triangle doc)", 
    default=False)

    gift_wrap : BoolProperty(name="Gift wrap", 
    description="Close the convex hull of the shape", 
    default=False)

    uv_mask : BoolProperty(name="UV mask", 
    description="Generate geometry only on parts enclosed in the UV quad. If False the whole texture is meshed in 3D space", 
    default=True)


### --- PANELS 

class TESS_PT_tesselate_UI(Panel):
    bl_label = "Tex plane tesselation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"
    bl_context = "objectmode" # only in object mode

    # @classmethod
    # def poll(cls, context):
    #     return context.object is not None # and context.mode == 'OBJECT'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        # Check for missing core dependencies (cv2 is required for all modes)
        if not module_can_be_imported('cv2'):
            layout.label(text="OpenCV required for basic functionality:", icon="ERROR")
            layout.label(text="cv2 module missing")
            layout.operator("preferences.addon_show", text="Open Preferences to Install", icon='PREFERENCES').module = __package__
            return
        
        # Triangle only needed for full tesselation
        triangle_missing = not module_can_be_imported('triangle')
        
        if context.object is None:
            layout.label(text='Select textured plane(s)')
            return

        if context.object is None:
            layout.label(text='Select textured plane(s)')
            return

        layout.prop(context.object, 'show_wire')

        # If triangle is missing, show a notice but allow contour_only mode
        if triangle_missing:
            layout.label(text="Triangle module not installed:", icon="INFO")
            layout.operator("preferences.addon_show", text="Open Preferences to Install", icon='PREFERENCES').module = __package__
            # layout.label(text="'Contour only' method available")

        row = layout.row()

        text = "Cut Texture Plane" if context.scene.ttp_props.contour_only else "Tesselate Texture Plane"
        row.operator("mesh.tesselate_plane", text=text, icon='MOD_TRIANGULATE')

        col = layout.column()

        # Show contour_only property
        col.prop(context.scene.ttp_props, "contour_only")
        # If triangle is missing, show a note
        if triangle_missing and not context.scene.ttp_props.contour_only:
            col.label(text="(Triangle missing : Required)", icon="INFO")
            col.separator()

        col.prop(context.scene.ttp_props, "simplify")
        
        scol = col.column()
        scol.enabled = not context.scene.ttp_props.contour_only
        scol.prop(context.scene.ttp_props, "aeration")
        col = layout.column()

        col.separator()
        col.prop(context.scene.ttp_props, "pix_margin")
        col.prop(context.scene.ttp_props, "pix_clean")

        scol = col.column()
        scol.enabled = not context.scene.ttp_props.contour_only
        scol.prop(context.scene.ttp_props, "external_only")
        col = layout.column()

        col.prop(context.scene.ttp_props, "uv_mask")


# Triangle sub-panel
class TESS_PT_subsettings_UI(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"

    bl_label = "Triangle extra settings"
    bl_parent_id = "TESS_PT_tesselate_UI"
    bl_options = {'DEFAULT_CLOSED'}#, 'HIDE_HEADER' 
    # bl_context = "objectmode" # only in object mode

    @classmethod
    def poll(cls, context):
        return context.object is not None and not context.scene.ttp_props.contour_only

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        
        layout.prop(context.scene.ttp_props, "min_angles")
        layout.prop(context.scene.ttp_props, "gift_wrap")
        layout.prop(context.scene.ttp_props, "true_delaunay")
        # layout.prop(context.scene.ttp_props, "algo_inc") # No need...
        # Maybe add a show wire option ?

class TTP_AddonPreferences(AddonPreferences):
    bl_idname = __package__
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Required Dependencies:")
        
        box = layout.box()
        col = box.column()
        
        # Check OpenCV (required for all functionality)
        row = col.row()
        has_cv2 = module_can_be_imported('cv2')
        if has_cv2:
            row.label(text="OpenCV (cv2): Installed", icon="CHECKMARK")
        else:
            row.label(text="OpenCV (cv2): Not Installed (Required for all modes)", icon="ERROR")
            row.operator("ttp.install_module", text="Install cv2").package_name = "opencv-python"
        
        # Check Triangle (required for full tesselation)
        row = col.row()
        has_triangle = module_can_be_imported('triangle')
        if has_triangle:
            row.label(text="Triangle: Installed", icon="CHECKMARK")
        else:
            row.label(text="Triangle: Not Installed (Required for tesselation mode)", icon="INFO")
            row.operator("ttp.install_module", text="Install triangle").package_name = "triangle"
            col.separator()
            col.label(text="A Blender restart is required after modules installation", icon="INFO")

        # Show available modes
        col.separator()
        col.label(text="Status:")
        if has_cv2 and has_triangle:
            col.label(text="All modules OK", icon="CHECKMARK")
        elif has_cv2:
            col.label(text="Limited functionality: Only 'Contour only' mode available", icon="ERROR")
        else:
            col.label(text="No functionality available: Install at least OpenCV", icon="CANCEL")

        # Help text
        layout.separator()
        col = layout.column()
        col.label(text="If installation fails, try running Blender as administrator")
        col.label(text="or install packages manually into your Blender modules folder:")
        col.label(text=str(Path(bpy.utils.user_resource('SCRIPTS', path='modules'))))


"""Manual Installation instructions
Try following solutions:
1. Try enabling addon after restarting blender as admin
2. If error is still there, try deleteting currently installed modules:
  - go to modules folder. Should be: {modules_loc}
  - delete folders "triangle", "cv2", "triangle...dist-infos", "opencv...dist-infos"
  - Try enabling the addon again to auto-install modules associated with this version of blender (preferably started as admin)
"""


class TTP_OT_InstallModule(Operator):
    bl_idname = "ttp.install_module"
    bl_label = "Install Module"
    bl_description = "Install the required module"
    
    package_name: StringProperty(
        name="Package Name",
        description="Name of the package to install"
    )
    
    def execute(self, context):
        import subprocess
        import sys
        try:
            # Get the user modules path and create it if needed
            user_modules = Path(bpy.utils.user_resource('SCRIPTS', path='modules', create=True))
            
            # Use subprocess to install the package
            python_exe = Path(sys.executable)
            cmd = [str(python_exe), "-m", "pip",  "--no-cache-dir", "install", 
                   f"--target={user_modules}", self.package_name, "--no-deps"]

            self.report({'INFO'}, f"Installing {self.package_name}...")
            
            # Run the installation command
            subprocess.check_call(cmd)
            
            self.report({'INFO'}, f"Successfully installed {self.package_name}")
            return {'FINISHED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Installation failed: {str(e)}")
            return {'CANCELLED'}

### --- REGISTER ---

classes = (
    TESS_props_group,
    TESS_OT_tesselate_plane,
    TESS_PT_tesselate_UI,
    TESS_PT_subsettings_UI,
    TTP_AddonPreferences,
    TTP_OT_InstallModule,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.ttp_props = PointerProperty(type=TESS_props_group)

def unregister():
    del bpy.types.Scene.ttp_props
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)