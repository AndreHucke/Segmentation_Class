# % Class to create interactive 3D VTK render window
# % EECE 8396: Medical Image Segmentation
# % Spring 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import vtk
import numpy as np

class vtkObject:
    def __init__(self, pnts=None, poly=None, actor=None):
        self.pnts = pnts
        self.poly = poly
        self.actor = actor

    def updateActor(self, verts):
        for j,p in enumerate(verts):
            self.pnts.InsertPoint(j,p)
        self.poly.Modified()


def ActorDecorator(func):
    def inner(verts,faces=None,
              colortable=None, coloridx=None,
              color=[1,0,0],opacity=1.0, pointSize=4, lineWidth=1):
        pnts = vtk.vtkPoints()
        for j,p in enumerate(verts):
            pnts.InsertPoint(j,p)

        poly = func(pnts,faces)

        #important for smooth rendering
        norm = vtk.vtkPolyDataNormals()
        norm.SetInputData(poly)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(norm.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if coloridx is None:
            actor.GetProperty().SetColor(color[0],color[1],color[2])
        else:
            scalars = vtk.vtkDoubleArray()
            for j in range(len(verts)):
                scalars.InsertNextValue(coloridx[j] / (len(colortable)-1))

            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(len(colortable))
            for j in range(len(colortable)):
                lut.SetTableValue(j,colortable[j,0],colortable[j,1], colortable[j,2])

            lut.Build()

            poly.GetPointData().SetScalars(scalars)
            norm.SetInputData(poly)
            mapper.SetInputConnection(norm.GetOutputPort())
            prop = actor.GetProperty()
            # prop.SetColor(0,0,0)
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange([0.0, 1.0])

        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetPointSize(pointSize)
        actor.GetProperty().SetLineWidth(lineWidth)
        obj = vtkObject(pnts, poly, actor)
        return obj

    return inner

@ActorDecorator
def pointActor(pnts, faces=None):
    cells = vtk.vtkCellArray()
    for j in range(pnts.GetNumberOfPoints()):
        vil = vtk.vtkIdList()
        vil.InsertNextId(j)
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetVerts(cells)

    return poly

@ActorDecorator
def linesActor(pnts,lines):
    cells = vtk.vtkCellArray()
    for j, f in enumerate(lines):
        vil = vtk.vtkIdList()
        vil.InsertNextId(lines[j,0])
        vil.InsertNextId(lines[j,1])
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetLines(cells)

    return poly

@ActorDecorator
def surfActor(pnts,faces):
    cells = vtk.vtkCellArray()
    for j, f in enumerate(faces):
        vil = vtk.vtkIdList()
        vil.InsertNextId(faces[j,0])
        vil.InsertNextId(faces[j,1])
        vil.InsertNextId(faces[j,2])
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetPolys(cells)

    poly.BuildCells()
    poly.BuildLinks()

    return poly



class myVtkWin(vtk.vtkRenderer):
    def __init__(self, sizex=512, sizey=512, title="3D Viewer (press q to quit)"):
        super().__init__()
        self.renwin = vtk.vtkRenderWindow() #creates a new window
        self.renwin.SetWindowName(title)
        self.renwin.AddRenderer(self)
        self.renwin.SetSize(sizex, sizey)
        self.inter = vtk.vtkRenderWindowInteractor() #makes the renderer interactive
        self.inter.AddObserver('KeyPressEvent',self.keypress_callback,1.0)
        self.lastpickpos = np.zeros(3)
        self.lastpickcell = -1
        self.inter.SetRenderWindow(self.renwin)
        self.inter.Initialize()
        self.inter.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.objlist = []

        self.renwin.Render() # paints the window on the screen once

    def __del__(self):
        del self.renwin, self.inter


    def addPoints(self, verts, color=[1.,0.,0.], opacity=1.):
        obj = pointActor(np.asarray(verts), color=color, opacity=opacity)
        self.objlist.append(obj)
        self.AddActor(obj.actor)
        if len(self.objlist)==1:
            mn = obj.actor.GetCenter()
            self.GetActiveCamera().SetFocalPoint(mn[0],mn[1],mn[2])

    def addLines(self, verts, lns, color=[1.,0.,0.], opacity=1., lineWidth=1):
        obj = linesActor(np.asarray(verts), np.asarray(lns), color=color,
                         opacity=opacity, lineWidth=lineWidth)
        self.objlist.append(obj)
        self.AddActor(obj.actor)
        if len(self.objlist)==1:
            mn = obj.actor.GetCenter()
            self.GetActiveCamera().SetFocalPoint(mn[0],mn[1],mn[2])

    def addSurf(self, verts, faces, color=[1.,0.,0.], opacity=1.,
                specular=0.9, specularPower=25.0, diffuse=0.6, ambient=0, edgeColor=None,
                colortable=None, coloridx=None):
        obj = surfActor(np.asarray(verts), np.asarray(faces), color=color, opacity=opacity, colortable=colortable, coloridx=coloridx)
        self.objlist.append(obj)
        actor = obj.actor
        if edgeColor is not None:
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(edgeColor[0], edgeColor[1], edgeColor[2])
        actor.GetProperty().SetAmbientColor(color[0], color[1], color[2])
        actor.GetProperty().SetDiffuseColor(color[0], color[1], color[2])
        actor.GetProperty().SetSpecularColor(1.0,1.0,1.0)
        actor.GetProperty().SetSpecular(specular)
        actor.GetProperty().SetDiffuse(diffuse)
        actor.GetProperty().SetAmbient(ambient)
        actor.GetProperty().SetSpecularPower(specularPower)

        self.AddActor(actor)
        if len(self.objlist)==1:
            mn = actor.GetCenter()
            self.GetActiveCamera().SetFocalPoint(mn[0],mn[1],mn[2])

    def keypress_callback(self,obj,ev):
        key = obj.GetKeySym()
        if (key == 'u' or key == 'U'):
            pos = obj.GetEventPosition()

            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.0005)

            picker.Pick(pos[0],pos[1],0,self)

            self.lastpickpos = picker.GetPickPosition()
            self.lastpickcell = picker.GetCellId()
        return key

    def updateActor(self, id, verts):
        self.objlist[id].updateActor(np.asarray(verts))

    def cameraPosition(self, position=None, viewup=None, fp=None , focaldisk=None):
        cam = self.GetActiveCamera()
        if position is not None:
            cam.SetPosition(position[0], position[1], position[2])
        if viewup is not None:
            cam.SetViewUp(viewup[0], viewup[1], viewup[2])
        if fp is not None:
            cam.SetFocalPoint(fp[0], fp[1], fp[2])
        if focaldisk is not None:
            dist  = np.sqrt(np.sum((np.array(cam.GetFocalPoint()) - np.array(cam.GetPosition()))**2))
            cam.SetFocalDisk(focaldisk*dist)

    def render(self):
        self.ResetCameraClippingRange()
        self.renwin.Render()
        self.inter.ProcessEvents()

    def start(self):
        self.inter.Start()