from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from PySide6 import QtCore, QtGui, QtWidgets
import torch

from graphfx.core import Graph, build_model_from_graph


@dataclass
class GNode:
    id: str
    type: str
    data: Dict[str, Any]
    pos: QtCore.QPointF


class NodeItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, gnode: GNode):
        super().__init__(-60, -30, 120, 60)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
        )
        self.setBrush(QtGui.QColor(240, 240, 240))
        self.setPen(QtGui.QPen(QtGui.QColor(80, 80, 80)))
        self.gnode = gnode
        self.setPos(gnode.pos)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(QtGui.QColor(20, 20, 20))
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, f"{self.gnode.type}\n{self.gnode.id}")


class EdgeItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, src: NodeItem, dst: NodeItem, label: str | None = None):
        super().__init__()
        self.src = src
        self.dst = dst
        self.label = label
        self.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 2))
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.update_path()

    def update_path(self):
        p = QtGui.QPainterPath()
        s = self.src.scenePos() + QtCore.QPointF(60, 0)
        t = self.dst.scenePos() + QtCore.QPointF(-60, 0)
        c1 = s + QtCore.QPointF(80, 0)
        c2 = t + QtCore.QPointF(-80, 0)
        p.moveTo(s)
        p.cubicTo(c1, c2, t)
        self.setPath(p)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        super().paint(painter, option, widget)
        if self.label:
            # Draw label at 50% of path length (approx by pointAtPercent)
            pt = self.path().pointAtPercent(0.5)
            painter.setPen(QtGui.QPen(QtGui.QColor(100, 60, 60)))
            painter.drawText(QtCore.QRectF(pt.x() - 30, pt.y() - 10, 60, 20), QtCore.Qt.AlignCenter, self.label)


class TempEdgeItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, src: NodeItem):
        super().__init__()
        self.src = src
        self.target_pos = src.scenePos()
        self.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 1, QtCore.Qt.DashLine))
        self.update_path()

    def set_target(self, pos: QtCore.QPointF):
        self.target_pos = pos
        self.update_path()

    def update_path(self):
        p = QtGui.QPainterPath()
        s = self.src.scenePos() + QtCore.QPointF(60, 0)
        t = self.target_pos
        c1 = s + QtCore.QPointF(80, 0)
        c2 = t + QtCore.QPointF(-80, 0)
        p.moveTo(s)
        p.cubicTo(c1, c2, t)
        self.setPath(p)


class GraphView(QtWidgets.QGraphicsView):
    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        zoom = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(zoom, zoom)


class PropertiesPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.form = QtWidgets.QFormLayout(self)
        self._editors: Dict[str, QtWidgets.QLineEdit] = {}
        self.current: GNode | None = None

    def set_node(self, node: GNode | None):
        for i in reversed(range(self.form.count())):
            item = self.form.itemAt(i)
            w = item.widget()
            if w:
                self.form.removeWidget(w)
                w.deleteLater()
        self._editors.clear()
        self.current = node
        if not node:
            return
        self.form.addRow(QtWidgets.QLabel(f"<b>{node.type} ({node.id})</b>"))
        for k, v in node.data.items():
            le = QtWidgets.QLineEdit(str(v))
            le.editingFinished.connect(self.changed.emit)
            self.form.addRow(k, le)
            self._editors[k] = le

    def read_back(self) -> Dict[str, Any]:
        if not self.current:
            return {}
        for k, le in self._editors.items():
            txt = le.text().strip()
            try:
                if "." in txt or "e" in txt.lower():
                    self.current.data[k] = float(txt)
                else:
                    self.current.data[k] = int(txt)
            except ValueError:
                self.current.data[k] = txt
        return self.current.data


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("graphfx - POC")
        self.resize(1000, 700)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = GraphView(self.scene)
        self.setCentralWidget(self.view)

        self.props = PropertiesPanel()
        self.props.changed.connect(self.on_props_changed)
        dock = QtWidgets.QDockWidget("Properties", self)
        dock.setWidget(self.props)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self.nodes: Dict[str, NodeItem] = {}
        self.edges: list[EdgeItem] = []
        self._data: Dict[str, Any] = {"nodes": [], "edges": [], "defaults": {}}

        self._build_menu()
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.scene.installEventFilter(self)
        self.view.setMouseTracking(True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

        self._edge_src: NodeItem | None = None
        self._temp_edge: TempEdgeItem | None = None

        self._load_default()

    def _build_menu(self):
        bar = self.menuBar()
        filem = bar.addMenu("File")
        open_act = filem.addAction("Open JSON…")
        save_act = filem.addAction("Save JSON…")
        run_act = filem.addAction("Run Model")
        open_act.triggered.connect(self.open_json)
        save_act.triggered.connect(self.save_json)
        run_act.triggered.connect(self.run_model)

        tb = QtWidgets.QToolBar("Tools", self)
        self.addToolBar(tb)
        add_embed = QtGui.QAction("Add Embedding", self)
        add_block = QtGui.QAction("Add Block", self)
        add_norm = QtGui.QAction("Add Norm", self)
        add_lm = QtGui.QAction("Add LMHead", self)
        tb.addAction(add_embed)
        tb.addAction(add_block)
        tb.addAction(add_norm)
        tb.addAction(add_lm)
        add_embed.triggered.connect(lambda: self.add_node("Embedding", {"vocab_size": 256, "hidden_size": 128}))
        add_block.triggered.connect(lambda: self.add_node("Block", {"hidden_size": 128, "num_heads": 4}))
        add_norm.triggered.connect(lambda: self.add_node("Norm", {"hidden_size": 128}))
        add_lm.triggered.connect(lambda: self.add_node("LMHead", {"hidden_size": 128, "vocab_size": 256}))

    def _load_default(self):
        try:
            with open("graphfx/example_graph.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_from_data(data)
        except Exception:
            pass

    def load_from_data(self, data: Dict[str, Any]):
        self.scene.clear()
        self.nodes.clear()
        self.edges.clear()

        x = 0
        for n in data.get("nodes", []):
            pos = n.get("position")
            qpos = QtCore.QPointF(pos["x"], pos["y"]) if isinstance(pos, dict) else QtCore.QPointF(x, 0)
            g = GNode(id=n["id"], type=n["type"], data=n.get("data", {}), pos=qpos)
            item = NodeItem(g)
            self.scene.addItem(item)
            self.nodes[g.id] = item
            x += 220

        for e in data.get("edges", []):
            src = self.nodes.get(e["source"]) if isinstance(e.get("source"), str) else None
            dst = self.nodes.get(e["target"]) if isinstance(e.get("target"), str) else None
            if src and dst:
                edge = EdgeItem(src, dst, e.get("label"))
                self.scene.addItem(edge)
                self.edges.append(edge)

        self._data = data

    def to_json(self) -> Dict[str, Any]:
        data = {"nodes": [], "edges": [], "defaults": self._data.get("defaults", {})}
        for nid, item in self.nodes.items():
            g = item.gnode
            g.pos = item.pos()
            data["nodes"].append({
                "id": g.id,
                "type": g.type,
                "data": g.data,
                "position": {"x": g.pos.x(), "y": g.pos.y()},
            })
        for e in self.edges:
            edge = {"source": e.src.gnode.id, "target": e.dst.gnode.id}
            if e.label:
                edge["label"] = e.label
            data["edges"].append(edge)
        return data

    def on_props_changed(self):
        self.props.read_back()

    def _on_selection_changed(self):
        items = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        self.props.set_node(items[0].gnode if items else None)

    def _tick(self):
        for e in self.edges:
            e.update_path()

    def open_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open graph JSON", "graphfx", "JSON (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_from_data(data)

    def save_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save graph JSON", "graphfx/graph.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    def run_model(self):
        g = Graph.from_json({
            "nodes": [{"id": nid, "type": item.gnode.type, "data": item.gnode.data} for nid, item in self.nodes.items()],
            "edges": [{"source": e.src.gnode.id, "target": e.dst.gnode.id, **({"label": e.label} if e.label else {})} for e in self.edges],
        })
        m = build_model_from_graph(g)
        vocab = int(self._data.get("defaults", {}).get("vocab_size", 256))
        x = torch.randint(0, vocab, (1, 16))
        attn = torch.ones(1, 16)
        out = m({"input_ids": x, "attention_mask": attn})
        last = list(out.keys())[-1]
        QtWidgets.QMessageBox.information(self, "Run", f"Last node: {last}\nShape: {tuple(out[last].shape)}")

    def add_node(self, ntype: str, data: Dict[str, Any]):
        base = {"Embedding": "embed", "Block": "blk", "Norm": "norm", "LMHead": "lm"}.get(ntype, ntype.lower())
        i = 1
        nid = f"{base}{i}"
        while nid in self.nodes:
            i += 1
            nid = f"{base}{i}"
        center = self.view.mapToScene(self.view.viewport().rect().center())
        g = GNode(id=nid, type=ntype, data=data, pos=center)
        item = NodeItem(g)
        self.scene.addItem(item)
        self.nodes[nid] = item

    def eventFilter(self, obj, event):
        if obj is self.scene:
            et = event.type()
            if et == QtCore.QEvent.GraphicsSceneMousePress:
                if event.modifiers() & QtCore.Qt.ControlModifier:
                    item = self.scene.itemAt(event.scenePos(), self.view.transform())
                    if isinstance(item, NodeItem):
                        self._edge_src = item
                        self._temp_edge = TempEdgeItem(item)
                        self.scene.addItem(self._temp_edge)
                        return True
            elif et == QtCore.QEvent.GraphicsSceneMouseMove:
                if self._temp_edge is not None:
                    self._temp_edge.set_target(event.scenePos())
                    return True
            elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
                if self._temp_edge is not None and self._edge_src is not None:
                    item = self.scene.itemAt(event.scenePos(), self.view.transform())
                    if isinstance(item, NodeItem) and item is not self._edge_src:
                        label = "mask" if (event.modifiers() & QtCore.Qt.ShiftModifier) else None
                        edge = EdgeItem(self._edge_src, item, label)
                        self.scene.addItem(edge)
                        self.edges.append(edge)
                    self.scene.removeItem(self._temp_edge)
                    self._temp_edge = None
                    self._edge_src = None
                    return True
            elif et == QtCore.QEvent.KeyPress:
                if event.key() == QtCore.Qt.Key_Delete:
                    self.delete_selected()
                    return True
                if event.key() == QtCore.Qt.Key_Escape:
                    if self._temp_edge is not None:
                        self.scene.removeItem(self._temp_edge)
                        self._temp_edge = None
                        self._edge_src = None
                        return True
        return super().eventFilter(obj, event)

    def delete_selected(self):
        for e in list(self.edges):
            if e.isSelected():
                self.scene.removeItem(e)
                self.edges.remove(e)
        for nid, item in list(self.nodes.items()):
            if item.isSelected():
                for e in list(self.edges):
                    if e.src is item or e.dst is item:
                        self.scene.removeItem(e)
                        self.edges.remove(e)
                self.scene.removeItem(item)
                del self.nodes[nid]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
