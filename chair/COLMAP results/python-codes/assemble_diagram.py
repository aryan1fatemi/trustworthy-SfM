"""Rebuild slue_feasible_set_diagram.svg: generated 3D panel A + unchanged panel B.

Panel A geometry comes from gen_panelA.build(); this module adds the labels and a
legend strip (as in the hand sketch), then splices the result around the existing
PANEL B section, which is copied through untouched.
"""
import os
import io
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_panelA as G

OUT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "feasible_sets", "slue_feasible_set_diagram.svg"))

W, H = 1060, 772
COL = {1: "#2b6cb0", 2: "#c05621", 3: "#2f855a"}
DARK = {1: "#1a365d", 2: "#7b341e", 3: "#22543d"}


def txt(x, y, body, size=14.5, fill="#1f2933", cls="it", anchor="start"):
    a = ' text-anchor="%s"' % anchor if anchor != "start" else ""
    return ('  <text x="%.1f" y="%.1f" font-size="%s" fill="%s" class="%s"%s>%s</text>'
            % (x, y, size, fill, cls, a, body))


def note(x, y, body, anchor="start"):
    a = ' text-anchor="%s"' % anchor if anchor != "start" else ""
    return '  <text x="%.1f" y="%.1f" class="note"%s>%s</text>' % (x, y, a, body)


def leader(p, q, colour="#9ca3af"):
    return ('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
            'stroke-width="0.9"/>' % (p[0], p[1], q[0], q[1], colour))


# --------------------------------------------------------------------------
def panel_a():
    frag, A = G.build(box=(38, 104, 584, 496))
    L = [frag]
    X = np.asarray(A["X"])

    # landmark
    L.append(leader((X[0] + 8, X[1] - 6), (X[0] + 32, X[1] - 24), "#1a365d"))
    L.append(txt(X[0] + 36, X[1] - 26,
                 'k &#8594; X&#770;<tspan class="sub">k</tspan>'
                 '<tspan font-style="normal"> = (x, y, z)</tspan>', 15.5, "#1a365d"))

    # short symbol labels only; meanings go in the legend
    off = {1: ((16, 20, "start"), (14, -8, "start")),
           2: ((-16, 6, "end"), (6, 20, "start")),
           3: ((0, 26, "middle"), (-10, -10, "end"))}
    for i in (1, 2, 3):
        c, y = np.asarray(A["c%d" % i]), np.asarray(A["y%d" % i])
        dc, dy = off[i]
        L.append(txt(c[0] + dc[0], c[1] + dc[1],
                     'c&#770;<tspan class="sub">%d</tspan>' % i, 15,
                     "#1f2933", anchor=dc[2]))
        L.append(txt(y[0] + dy[0], y[1] + dy[1],
                     'y<tspan class="sub">%dk</tspan>' % i, 13.5, DARK[i],
                     anchor=dy[2]))

    # name one pyramid and one tolerance box
    mid1 = (np.asarray(A["c1"]) + X) / 2
    L.append(leader((mid1[0] + 22, mid1[1] - 46), (mid1[0] - 6, mid1[1] - 16), COL[1]))
    L.append(txt(mid1[0] + 26, mid1[1] - 50,
                 'P<tspan class="sub">1k</tspan>'
                 '<tspan class="note"> &#8594; rectangular pyramid</tspan>',
                 15, COL[1]))

    b2 = np.asarray(A["box2"]).mean(0)
    L.append(leader((b2[0] + 6, b2[1] + 8), (b2[0] + 44, b2[1] + 42), COL[2]))
    L.append(txt(b2[0] + 48, b2[1] + 47,
                 'X<tspan class="sub">2k</tspan>'
                 '<tspan class="note"> &#8594; square pixel box</tspan>', 14, COL[2]))
    pl2 = np.asarray(A["plane2"])
    L.append(note(pl2[2][0] + 10, pl2[2][1] + 2, "image plane"))

    # feasible set callout
    L.append('  <circle cx="%.1f" cy="%.1f" r="21" fill="none" stroke="#9ca3af" '
             'stroke-width="1.1" stroke-dasharray="4 3"/>' % (X[0], X[1]))
    L.append(leader((X[0] - 18, X[1] - 12), (X[0] - 74, X[1] - 48), "#1a365d"))
    L.append(txt(X[0] - 78, X[1] - 66,
                 '&#8497;<tspan class="sub">k</tspan>'
                 '<tspan font-style="normal"> = &#8898;</tspan>'
                 '<tspan class="sub" font-style="normal">i</tspan>'
                 '<tspan>&#8194;P</tspan><tspan class="sub">ik</tspan>', 15.5, "#1a365d",
                 anchor="end"))
    L.append(note(X[0] - 78, X[1] - 50, "feasible set / containment volume", "end"))
    return "\n".join(L)


def legend(x=38, y=516, w=546):
    rows = [
        ('c&#770;<tspan class="sub">i</tspan>', "centre of camera i, the apex of the cone"),
        ('&#952;&#770;<tspan class="sub">i</tspan>'
         '<tspan font-style="normal"> = [ </tspan>R&#770;<tspan class="sub">i</tspan>'
         '<tspan font-style="normal"> | </tspan>t&#770;<tspan class="sub">i</tspan>'
         '<tspan font-style="normal"> ]</tspan>', "camera pose, 6 DoF"),
        ('y<tspan class="sub">ik</tspan>'
         '<tspan font-style="normal"> = ( </tspan>u<tspan class="sub">ik</tspan>'
         '<tspan font-style="normal">, </tspan>v<tspan class="sub">ik</tspan>'
         '<tspan font-style="normal"> )</tspan>', "detected keypoint in view i"),
        ('X<tspan class="sub">ik</tspan>'
         '<tspan font-style="normal"> = [ </tspan>u<tspan class="sub">ik</tspan>'
         '<tspan font-style="normal"> &#177; &#949; ] &#215; [ </tspan>'
         'v<tspan class="sub">ik</tspan><tspan font-style="normal"> &#177; &#949; ]</tspan>',
         "square pixel tolerance box"),
        ('P<tspan class="sub">ik</tspan>', "back-projection pyramid of that box"),
        ('X<tspan class="sub">c</tspan>'
         '<tspan font-style="normal"> = </tspan>R&#770;<tspan class="sub">i</tspan>'
         '<tspan font-style="normal"> ( X &#8722; </tspan>c&#770;<tspan class="sub">i</tspan>'
         '<tspan font-style="normal"> )</tspan>', "camera-frame coordinates"),
    ]
    out = ['  <rect x="%d" y="%d" width="%d" height="120" rx="4" fill="#fafafa" '
           'stroke="#e5e7eb" stroke-width="1"/>' % (x, y, w)]
    for n, (sym, mean) in enumerate(rows):
        col, row = divmod(n, 3)
        px = x + 16 + col * (w / 2 - 4)
        py = y + 28 + row * 32
        out.append(txt(px, py, sym, 13.5))
        out.append(note(px, py + 16, mean))
    return "\n".join(out)


def footer(y=680):
    return "\n".join([
        '  <line x1="28" y1="%d" x2="1032" y2="%d" stroke="#e5e7eb" '
        'stroke-width="1"/>' % (y, y),
        txt(28, y + 30,
            '&#8497;<tspan class="sub">k</tspan>'
            '<tspan font-style="normal"> = { X &#8712; &#8477;&#179; </tspan>|'
            '<tspan font-style="normal"> &#8214; &#960;( </tspan>'
            '&#952;&#770;<tspan class="sub">i</tspan>'
            '<tspan font-style="normal">, X ) &#8722; </tspan>'
            'y<tspan class="sub">ik</tspan><tspan font-style="normal"> &#8214;</tspan>'
            '<tspan class="sub" font-style="normal">&#8734;</tspan>'
            '<tspan font-style="normal"> &#8804; &#949;   for all </tspan>'
            'i<tspan font-style="normal"> &#8712; views(</tspan>k'
            '<tspan font-style="normal">) }</tspan>', 14.5),
        note(28, y + 54,
             'Cameras are held fixed at their estimates &#952;&#770;, so every '
             'constraint is linear in X and &#8497;'
             '<tspan class="sub" font-style="normal">k</tspan>'
             '<tspan> is a convex polyhedron.</tspan>'),
    ])


_HAT_TUNE = {"X": (-0.88, -0.44, 0.10), "R": (-0.88, -0.44, 0.10),
             "c": (-0.52, -0.40, -0.18), "t": (-0.46, -0.50, -0.14),
             "&#952;": (-0.62, -0.48, -0.10)}


def fix_hats(svg):
    """Replace <letter>&#770; with a manually positioned circumflex.

    Georgia (and Times) do not compose the combining circumflex over capitals,
    so X&#770; renders as "X ^".  This draws the hat as a shifted tspan and
    restores the pen with a hair space, which composes correctly everywhere.
    """
    for ch, (dx, dy, back) in _HAT_TUNE.items():
        svg = svg.replace(
            ch + "&#770;",
            '%s<tspan font-size="0.70em" dx="%gem" dy="%gem">&#710;</tspan>'
            '<tspan dx="%gem" dy="%gem">&#8202;</tspan>' % (ch, dx, dy, back, -dy))
    return svg


def _drop_close(s):
    """Remove a trailing </svg> without eating neighbouring characters."""
    s = s.rstrip()
    return s[:-len("</svg>")].rstrip() if s.endswith("</svg>") else s


def main():
    src = io.open(OUT, encoding="utf-8").read()
    head = src.split("<!-- ============ PANEL A ============ -->", 1)[0]
    rest = src.split("<!-- ============ PANEL B ============ -->", 1)[1]
    panel_b = rest.split("<!-- ============ FOOTER ============ -->", 1)[0] \
        if "FOOTER" in rest else rest.rsplit("<line x1=\"28\"", 1)[0]

    head = re.sub(r'viewBox="0 0 \d+ \d+" width="\d+" height="\d+"',
                  'viewBox="0 0 %d %d" width="%d" height="%d"' % (W, H, W, H), head)
    head = re.sub(r'<rect width="\d+" height="\d+" fill="#ffffff"/>',
                  '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H), head)

    panel_b = panel_b.replace(
        '<text x="622" y="68" class="sh">long along the weak direction</text>',
        '<text x="622" y="68" class="sh">long along the weak direction '
        '&#8212; a zoom on the dashed circle in A</text>')

    body = (head
            + "<!-- ============ PANEL A ============ -->\n"
            + '  <text x="28" y="32" class="hd">A. Three back-projection pyramids '
              'and their intersection</text>\n'
            + '  <text x="28" y="52" class="sh">each pixel tolerance box back-projects '
              'to a 4-sided pyramid with apex at the camera centre;</text>\n'
            + '  <text x="28" y="68" class="sh">the opening is drawn hugely '
              'exaggerated &#8212; &#949; is about one pixel out of 800</text>\n\n'
            + panel_a() + "\n\n"
            + legend() + "\n\n"
            + "  <!-- ============ PANEL B ============ -->"
            + _drop_close(panel_b) + "\n\n"
            + "  <!-- ============ FOOTER ============ -->\n"
            + footer() + "\n</svg>\n")

    io.open(OUT, "w", encoding="utf-8").write(fix_hats(body))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
