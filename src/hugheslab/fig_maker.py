# -*- coding: utf-8 -*-
"""
Functions to help build nice figures.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
import io
import string
import html
import re
import xml.etree.ElementTree as ET
from svgutils.compose import Figure, Panel, SVG
from svgutils.transform import fromstring

import numpy as np

from PIL import Image
    


def img_scalebar(img, length, orientation = 'horizontal', pixel_value = 255, width = 20, pixel_size = 1, border = 20):
    """ Adds a scalebar to an image by directly writing pixel values.

    Arguments:
        img:  ndarray
              the original image, 2D or 3D numpy array
        length: int
                length of scalebar in real units
        
    Keyword Arguments:
        orientation: str
                     'horizontal' or 'vertical' (default 'horizontal')
        pixel_value: int
                     pixel value to use for scalebar (default 255)
        width: int
               width of scalebar in pixels (default 20)
        pixel_size: float
                    size of a pixel in real units (default 1)
        border: int
                distance from edge of image in pixels (default 20)

    Returns:
        img: ndarray
             the image with scalebar added
    """         
                          
        
    
    if img.ndim == 3:
        n, h, w = img.shape
    else:
        h,w = img.shape
        
    len_px = int(round(length / pixel_size))
    
    if orientation == 'horizontal':
        x1 = w - len_px - border
        x2 = w - border
        y1 = h - width - border
        y2 = h - border
        
       
    elif orientation == 'vertical':
        x1 = w - width - border
        x2 = w - border
        y1 = h - len_px - border
        y2 = h - border

    if img.ndim == 3:
        img[:,y1:y2, x1:x2] = pixel_value
    else:
        img[y1:y2, x1:x2] = pixel_value
    
    return img
    
    


def fig_to_svg_io(fig):
    """
    Convert a matplotlib figure to an in-memory SVG buffer.
    """
    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="svg",
        bbox_inches="tight",
    )

    buf.seek(0)
    return buf

def combine_figs(
    figures,
    ncols=2,
    panel_size=None,
    scale=1.0,
    white_background=True,
    labelLetters=True,
    labels=None,
    rowLabels=None,
    columnLabels=None,
    fontSize=12,
    fontFamily="Times New Roman",
):
    """
    Combine matplotlib figures into a single SVG layout.

    Set ``panel_size=None`` to auto-pack using each SVG's actual size,
    with per-column widths and per-row heights.
    Set ``white_background=True`` to add an opaque white canvas.
    Label options mirror ``multi_im``.
    """

    n = len(figures)
    nrows = math.ceil(n / ncols)

    roots = []
    widths = []
    heights = []

    for fig in figures:
        svg_str = fig_to_svg_str(fig)
        svg = fromstring(svg_str)
        root = svg.getroot()
        fig_w, fig_h = svg.get_size()
        fig_w, fig_h = _svg_intrinsic_size(svg_str, fig_w, fig_h)

        roots.append(root)
        widths.append(fig_w * scale)
        heights.append(fig_h * scale)

    if panel_size is None:
        col_widths = [0.0] * ncols
        row_heights = [0.0] * nrows

        for idx in range(n):
            row = idx // ncols
            col = idx % ncols
            col_widths[col] = max(col_widths[col], widths[idx])
            row_heights[row] = max(row_heights[row], heights[idx])
    else:
        panel_w, panel_h = panel_size
        col_widths = [panel_w * scale] * ncols
        row_heights = [panel_h * scale] * nrows

    x_offsets = [0.0]
    for w in col_widths[:-1]:
        x_offsets.append(x_offsets[-1] + w)

    total_w = sum(col_widths)
    total_h = sum(row_heights)

    has_panel_labels = labelLetters or (labels is not None)
    top_margin = fontSize * 3.0 if columnLabels is not None else 0.0
    left_margin = fontSize * 4.0 if rowLabels is not None else 0.0
    panel_label_band = fontSize * 3.0 if has_panel_labels else 0.0

    row_pitches = [h + panel_label_band for h in row_heights]

    y_offsets = [0.0]
    for pitch in row_pitches[:-1]:
        y_offsets.append(y_offsets[-1] + pitch)

    total_h_with_row_labels = sum(row_pitches)

    out_w = total_w + left_margin
    out_h = top_margin + total_h_with_row_labels

    h_offsets = [left_margin + x for x in x_offsets]
    v_offsets = [top_margin + y for y in y_offsets]

    panels = []
    for idx, root in enumerate(roots):
        row = idx // ncols
        col = idx % ncols

        root.scale(scale)
        root.moveto(h_offsets[col], v_offsets[row])
        panels.append(root)

        if has_panel_labels:
            txt = ''
            if labelLetters:
                txt += f"({string.ascii_lowercase[idx]})"
            if labels is not None and idx < len(labels):
                txt += labels[idx]

            if txt:
                label_x = h_offsets[col] + (col_widths[col] / 2.0)
                label_y = v_offsets[row] + row_heights[row] + (panel_label_band * 0.45)
                panels.append(
                    _svg_text_element(
                        out_w,
                        out_h,
                        label_x,
                        label_y,
                        txt,
                        fontSize,
                        fontFamily,
                        anchor='middle',
                    )
                )

    if rowLabels is not None:
        for row_idx in range(min(len(rowLabels), nrows)):
            txt = rowLabels[row_idx]
            label_x = fontSize * 1.6
            label_y = v_offsets[row_idx] + (row_heights[row_idx] / 2.0)
            panels.append(
                _svg_text_element(
                    out_w,
                    out_h,
                    label_x,
                    label_y,
                    txt,
                    fontSize,
                    fontFamily,
                    anchor='middle',
                    rotate=-90,
                )
            )

    if columnLabels is not None:
        for col_idx in range(min(len(columnLabels), ncols)):
            txt = columnLabels[col_idx]
            label_x = h_offsets[col_idx] + (col_widths[col_idx] / 2.0)
            label_y = fontSize * 1.8
            panels.append(
                _svg_text_element(
                    out_w,
                    out_h,
                    label_x,
                    label_y,
                    txt,
                    fontSize,
                    fontFamily,
                    anchor='middle',
                )
            )

    if white_background:
        bg_svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{out_w}" height="{out_h}" '
            f'viewBox="0 0 {out_w} {out_h}">'
            f'<rect x="0" y="0" width="{out_w}" height="{out_h}" fill="white"/>'
            f'</svg>'
        )
        panels.insert(0, fromstring(bg_svg).getroot())

    return Figure(
        str(out_w),
        str(out_h),
        *panels,
    )



def fig_to_svg_str(fig):
    """
    Convert matplotlib figure to SVG string.
    """
    buf = io.StringIO()

    fig.savefig(
        buf,
        format="svg",
        bbox_inches="tight",
    )

    return buf.getvalue()


def _svg_length_to_px(length):
    """
    Parse an SVG length string (e.g. '420pt', '600px') into px.
    """
    if isinstance(length, (int, float)):
        return float(length)

    text = str(length).strip()
    match = re.match(r'^([0-9]+(?:\.[0-9]+)?)([a-z%]*)$', text)
    if match is None:
        return float(text)

    value = float(match.group(1))
    unit = match.group(2).lower()

    if unit in ('', 'px'):
        return value
    if unit == 'pt':
        return value * (96.0 / 72.0)
    if unit == 'in':
        return value * 96.0
    if unit == 'cm':
        return value * (96.0 / 2.54)
    if unit == 'mm':
        return value * (96.0 / 25.4)

    return value


def _svg_intrinsic_size(svg_str, fallback_w, fallback_h):
    """
    Get SVG drawing-space size from viewBox when available.
    """
    try:
        root = ET.fromstring(svg_str)
        view_box = root.attrib.get('viewBox', '')
        parts = view_box.replace(',', ' ').split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    except Exception:
        pass

    return _svg_length_to_px(fallback_w), _svg_length_to_px(fallback_h)


def _svg_text_element(canvas_w, canvas_h, x, y, txt, font_size, font_family, anchor='start', rotate=None):
    """
    Create an SVG text element as a root node suitable for svgutils Figure.
    """
    transform = ''
    if rotate is not None:
        transform = f' transform="rotate({rotate} {x} {y})"'

    text_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_w}" height="{canvas_h}" '
        f'viewBox="0 0 {canvas_w} {canvas_h}">'
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{html.escape(str(font_family), quote=True)}" '
        f'font-size="{font_size}"{transform}>'
        f'{html.escape(str(txt))}'
        f'</text>'
        f'</svg>'
    )
    return fromstring(text_svg).getroot()


def multi_im(imgs, nCols = 4, labelLetters = True, labels = None, rowLabels = None, columnLabels = None, fontSize = 8, fontFamily = "Times New Roman", width = 140, height = None, dpi = 150, autoScale = True):
    """ Produces a multi-panel figure made up of images provided as numpy arrays.
    
    Arguments:
        img          : list of ndarray
        list of images, each image as 2D/3D np array
        
    Optional Keyword Arguments:        
        nCols        : int
                       number of columns in figure (default is 4)
        labelLetters : boolean
                       if true, each image gets a letter label (default is True)
        labels       : list of str
                       list of labels to use for images (default None).
        rowLabels    : list of str
                       list of labels for rows (default None)
        columnLabels : list of str
                       list of labels for columns (default None)
        fontSize     : float
                       font size for all labels
        fontFamily   : str
                       font family as a string
        width        : float
                       width of figure in mm (default 140)
        height       : float or None
                       height of figure in mm (if None, calculate based on square images)
        dpi          : float
                       dpi of figure (defaut 150)
        
    Returns :
        Matplotlib figure
        
    """
    
    plt.rcParams["font.family"] = fontFamily
    plt.rcParams["font.size"] = fontSize

    nIms = len(imgs)

    nRows = math.ceil(nIms / nCols)
    
    width = width / 25.4                # convert mm to inches
    
    if height is None:
        height = (width / nCols * nRows)
    else:
        height = height / 25.4
        
    fig, axes = plt.subplots(nRows, nCols, dpi=dpi, figsize = (width, height), sharex=False, sharey=False)

    for idx, img in enumerate(imgs):
        txt = ''
        if nRows > 1 and nCols > 1:            
            this = axes[idx // nCols, idx % nCols ]
        else:
            this = axes[idx]
        if autoScale:
            this.imshow(img, cmap='gray')
        else:
            this.imshow(img, cmap='gray', vmin = 0, vmax = 1)
        if labelLetters:
            txt = txt + f"({string.ascii_lowercase[idx]})"
        if labels is not None:
            txt = txt + labels[idx]
        this.set_xlabel(txt) 
   

    for x, axi in enumerate(axes):
        if nRows > 1 and nCols > 1:
            for y, ax in enumerate(axi):
                 if x * nCols + y > nIms - 1:
                     ax.set_axis_off()
                 ax.set_yticks([])
                 ax.set_xticks([])
        else:
            if x > nIms - 1:
                axi.set_axis_off()
            axi.set_yticks([])
            axi.set_xticks([])
            
            
    if rowLabels is not None:
        for idx in range(len(rowLabels)):
            fig.get_axes()[idx * nCols].set_ylabel(rowLabels[idx])   
            
    if columnLabels is not None:
        for idx in range(len(columnLabels)):
            fig.get_axes()[idx].set_title(columnLabels[idx], fontsize = fontSize)           
       
 
             
    return fig    


def img_zoom(img, loc, zoom = 4, place = 'se', displayLoc = None):
    """ Adds an zoomed inset to an image.
    
    Arguments:
        img    : ndarray
                 the original image, 2D numpy array
        loc    : tuple
                 location of inset, tuple of (x,y,w,h)
        zoom   : float
                 zoom factor (default 4)
        place  : str
                 compass location: 'se', 'ne', 'nw' or 'sw'
    
    """
    
    zoomImg = get_zoom(img, loc, zoom)
    
    zy, zw = np.shape(zoomImg)

    if place == 'se':
        img[-zy:,-zw:] = zoomImg
    elif place == 'ne':
        img[:zy,-zw:] = zoomImg
    elif place == 'nw':
        img[:zy, :zw] = zoomImg    
    elif place == 'sw':
        img[-zy:,:zw] = zoomImg
    return img


def img_ax(ax):
    """ Returns image data from axis as numpy array
    """
    
    return ax.get_images()[0].get_array().toflex()['_data'] 
    


def ax_zoom(ax, loc, zoom = 4, place = 'se', 
            displayLoc = None, offset = 0,
            originBox = True, zoomBox = True,
            lines = True, mainBox = True, autoScale = True, scaleFactor = 1):
    """ Adds a zoom inset to an axis
    
    Arguments:
        ax         : reference to axis
        loc        : zoomed area as tuple of (x,y,w,h)
        
    Keyword Arguments:
        zoom       : zoom factor
        place      : reserved
        displayLoc : reserved
        offset     : overhang of inset over edge of original image, either int or tuple of (x,y)
        originBox  : True to draw box around zoom in original image (default)
        zoomBox    : True to draw box around zoom in zoomed image (default)
        lines      : True to draw lines between original and new areas (default)
        mainBox    : True to frame image (default)
    
    """
   
    img = img_ax(ax)
    
    h,w = np.shape(img)[:2]
    if isinstance(offset,tuple):
        xoffset, yoffset = offset
    else:
        xoffset = offset
        yoffset = offset

    xoffset = int(xoffset * w)
    yoffset = int(yoffset * w)

    ox,oy,ow,oh = loc

    zoomImg = get_zoom(img, loc, zoom) * scaleFactor

    if img.ndim == 3:
        imgOut = max(np.max(img), np.max(zoomImg)) * np.ones((h + yoffset, w + xoffset, np.shape(img)[2]), dtype = img.dtype)
        imgOut[:h, :w, :] = img
    else:
        imgOut = max(np.max(img), np.max(zoomImg)) * np.ones((h + yoffset, w + xoffset), dtype = img.dtype)
        imgOut[:h, :w] = img
        

    zh, zw = np.shape(zoomImg)[:2]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
   
    y = h + xoffset - zh
    x = w + yoffset- zw
    if img.ndim == 3:
        imgOut[-zh:,-zw:,:] = zoomImg
    else:
        imgOut[-zh:,-zw:] = zoomImg
    if autoScale:
        ax.imshow(imgOut, cmap = ax.get_images()[0].get_cmap())
    else:
        ax.imshow(imgOut, cmap = ax.get_images()[0].get_cmap(), vmin = 0, vmax = 1)

        

    if mainBox:
        ax.add_patch(patches.Rectangle((0, 0), 0, h, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((0, 0), w, 0, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((0, h), min(w + xoffset - zw, w), 0, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((w, 0), 0, min(h + yoffset - zh,h), linewidth=1, edgecolor='k', facecolor='none'))
        
    
    # inset
    if zoomBox:
        ax.add_patch(patches.Rectangle((x, y), zw, zh, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((x, y), zw, zh, linewidth=1, edgecolor='w', ls = '--', facecolor='none'))

    # origin
    if originBox:
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, linewidth=1, edgecolor='w', ls = '--', facecolor='none'))

    # connecting lines
    if lines and originBox and zoomBox:
        fh,fw = np.shape(imgOut)[:2]
    
        ax.add_patch(patches.Polygon(np.array(((ox, oy + oh), (fw - zw, fh))), closed = False, linewidth=1, edgecolor='k',  facecolor='none'))
        ax.add_patch(patches.Polygon(np.array(((ox + ow, oy), (fw , fh - zh))), closed = False, linewidth=1, edgecolor='k', facecolor='none'))
        ax.add_patch(patches.Polygon(np.array(((ox, oy + oh), (fw - zw, fh))), closed = False, linewidth=1, edgecolor='w',  ls = '--', facecolor='none'))
        ax.add_patch(patches.Polygon(np.array(((ox + ow, oy), (fw , fh - zh))), closed = False, linewidth=1, edgecolor='w', ls = '--', facecolor='none'))


def scalebar(ax, length = 100, text = None, hoffset = 1, voffset = 2, w = None, h = 1, place = 'nw', padding = 1, shadow = True):
    """ Adds a scalebar to an image
    
    Arguments:
        ax      : axis object
        length  : int
                  length of scalebar in image pixels
        text    : str
                  label to sit next to scale bar
        hoffset : float
                  distance from horizontal edge
        voffset : float
                  distance from vertiacl end
        w       : float
                  if specified sets widths of bar to this in mm
        h       : float
                  height of bar in mm
        place   : str
                  compass point location, 'nw' or 'se' 
    """
    
   
    hoffset = mm2pix(ax,hoffset)
    voffset = mm2pix(ax,voffset)
    y = voffset + min(ax.get_ylim())
    x = hoffset + min(ax.get_xlim())
    
    imH, imW = np.shape(img_ax(ax))[:2]
    
    h = mm2pix(ax,h)

    if w is not None:
        w = mm2pix(ax,w)
    else:
        w = length
    
    if place == 'nw':
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='w', facecolor='w'))
    elif place == 'se':
        ax.add_patch(patches.Rectangle((imW - x - w, imH - y - h), w, h, linewidth=1, edgecolor='w', facecolor='w'))

        
    if text is not None:
        r = ax.get_figure().canvas.get_renderer()

        if place == 'nw':

            tx = x + w + hoffset
            ty = y
            t = ax.text(tx, ty , text, va = 'center_baseline', fontsize = 'small', color = 'w') #, bbox={"fill": True, "lw": 0.2, "color": (0,0,0,.5),"boxstyle": "square,pad=0.1"})
    
            bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            tw = bb.width
            th = bb.height
            padding = mm2pix(ax,padding)
            if shadow: ax.add_patch(patches.Rectangle((x - padding, y + th/2 - padding), tx - x + tw + padding *2, -th + padding *2, linewidth=1, edgecolor=None, ls = '--', facecolor=(0,0,0,0.5)))
            if shadow: ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='w', facecolor='w'))
        
        elif place == 'se':
            
            tx = imW - x - w - hoffset
            ty = imH - y - h
            t = ax.text(tx, ty , text, va = 'center_baseline', ha = 'right', fontsize = 'small', color = 'w') #, bbox={"fill": True, "lw": 0.2, "color": (0,0,0,.5),"boxstyle": "square,pad=0.1"})
    
            bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            tw = bb.width
            th = bb.height
            padding = mm2pix(ax,padding)
            if shadow: ax.add_patch(patches.Rectangle((imW - x - w - tw - hoffset - padding, imH - y + th ), tx + tw + padding *2, -th + padding *2, linewidth=1, edgecolor=None, ls = '--', facecolor=(0,0,0,0.5)))
            if shadow: ax.add_patch(patches.Rectangle((imW - x - w, imH - y - h), w, h, linewidth=1, edgecolor='w', facecolor='w'))
         
            

def mm2pix(ax, mm):
    """ Converts a dimensions in mm to pixels for a specified axis
    """
    
    dpi = ax.get_figure().get_dpi()
    imWidth = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    imPhysicalWidth = ax.get_window_extent().width / dpi * 25.4
    
    frac = mm / imPhysicalWidth 
    
    return frac * imWidth

    

def get_zoom(img, loc, zoom = 4):
    """ Gets a zoomed image from part of another image.
    """
    x,y,w,h = loc
    
    roi = img[y:y + h, x:x+w]
    im = Image.fromarray(roi)
    zoom = np.array(im.resize((int(w * zoom), int(h * zoom))))
    
    return zoom



def inches(i1, i2):
    """ Converts mm to inches, for use with figure size in matplotlib   
    """
    
    return i1 / 25.4, i2 / 25.4
