import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings


from orangecontrib.spectroscopy.data import _spectra_from_image, getx, build_spec_table
from orangecontrib.spectroscopy.utils import get_hypercube


def get_hypercubes(images):
    output = []
    lsx, lsy = None, None
    for im in images:
        hypercube, lsx, lsy = get_hypercube(im, im.domain["map_x"], im.domain["map_y"])
        output.append(hypercube)
    return output, lsx, lsy


def compute_theta(images):
    return 0.5 * np.arctan2(images[1] - images[3], images[0] - images[2])


def compute_intensity(images):
    S0 = (images[0] + images[1] + images[2] + images[3]) * 0.5
    return S0


def compute_amp(images):
    return np.sqrt((images[3] - images[1])**2 + (images[2] - images[0])**2) / compute_intensity(images)


def hypercube_to_table(hc, wns, lsx, lsy):
    table = build_spec_table(*_spectra_from_image(hc,
                             wns,
                             np.linspace(*lsx),
                             np.linspace(*lsy)))
    return table


def process_polar(images):
    hypercubes, lsx, lsy = get_hypercubes(images)

    wns = getx(images[0])

    th = compute_theta(hypercubes)
    amp = compute_amp(hypercubes)
    int = compute_intensity(hypercubes)

    # join absorbance from images into a single image with a mean
    intensity = hypercube_to_table(int, wns, lsx, lsy)
    tht = hypercube_to_table(th, wns, lsx, lsy)
    ampt = hypercube_to_table(amp, wns, lsx, lsy)

    output = intensity
    output.th = tht
    output.amp = ampt

    return output, intensity, tht, ampt


class OWPolar(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Polar"

    # Short widget description
    description = (
        "Polar.")

    icon = "icons/unknown.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        polar = Output("Polar Data", Orange.data.Table, default=True)
        intensity = Output("Intensity", Orange.data.Table)
        theta = Output("Theta", Orange.data.Table)
        amplitude = Output("Amplitude", Orange.data.Table)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")


    @Inputs.data
    def set_data(self, dataset):
        self.data = dataset
        self.commit()

    def commit(self):
        if self.data is None:
            return

        # TODO for now this assumes images in the correct order of filenames with a Filename column

        fncol = self.data[:, "Filename"].metas.reshape(-1)
        unique_fns = np.unique(fncol)

        # split images into separate tables
        images = []
        for fn in unique_fns:
            images.append(self.data[fn == fncol])

        # TODO align images according to their positions

        out, int, th, amp = process_polar(images)
        self.Outputs.polar.send(out)
        self.Outputs.intensity.send(int)
        self.Outputs.theta.send(th)
        self.Outputs.amplitude.send(amp)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPolar).run(Orange.data.Table("/home/marko/polar_preprocessed.pkl.gz"))
