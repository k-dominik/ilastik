import logging
import typing

from ilastik.shell.server.slotApi import WrappeSlot
from ilastik.applet import Applet
from lazyflow.graph import InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper


logger = logging.getLogger(__name__)


class WrappedApplet(object):
    def __init__(self, applet: Applet) -> None:
        """

        Attributes:
          name: str
          applet_type: typing.Type[Applet]
          input_slots: typing.List[InputSlot]
          output_slots: typing.List[OutputSlot]


        """
        self._applet = None
        self._input_slots = None
        self._output_slots = None

        self._initialize_applet(applet)

    def _initialize_applet(self, applet: Applet, forced_axes: str='tczyx') -> None:
        tlo = applet.topLevelOperator
        # TODO: initialize inputs and outputs
        for input_slot in tlo.inputSlots:
            logger.debug(f"adding input_slot: {input_slot}")
            # TODO: need to check if connected?!
            self._input_slots[input_slot.name] = {
                'slot': input_slot
            }

        for output_slot in tlo.outputSlots:
            logger.debug(f"adding output_slot: {output_slot}")

            op_reorder.AxisOrder.setValue(forced_axes)
            op_reorder.Input.connect(output_slot)
            self._output_slots[output_slot.name] = {
                '_slot': output_slot,
                'slot': OpReorderAxes(op_reorder.Output)

            }

    @property
    def input_slots(self):
        return self._input_slots

    @property
    def output_slots(self):
        return self._output_slots


class Applets(object):
    def __init__(self, applets: typing.List[Applet]) -> None:
        self._applets: typing.List[Applet] = []
        self._initialize_applets(applets)

    def _initialize_applets(self, applets: typing.List[Applet]) -> None:
        for applet in applets:
            self._initialize_applet(applet)

    def _initialize_applet(self, applet: Applet) -> None:
        # TODO: build and index or something, or memoize getitem
        # TODO: register callbacks
        self._applets.append(WrappedApplet(applet))

    def __getitem__(
            self,
            key: typing.Union[int, str, typing.Type[Applet]]
        ) -> typing.Union[Applet, typing.List[Applet]]:
        """Return applet object

        Args:
            key (int|str|Applet): returns the applet matching the key. Depending
              on the key type, different behaviour is implemented:
              int: simply returns a (single) Applet at index key
              str: returns Applet, or list of Applets with Applet.name matching
                key exactly. List is returns with mutltiple matches
                Applet: returns Applet, or list of Applets
        """
        pass

    def _get_applet_by_index(self, index: int) -> Applet:
        return self._applets[index]

    def _get_applet_by_name(self, name: str) -> typing.Union[Applet, typing.List[Applet]]:
        matches = [applet for applet in self._applets if applet.name == name]
        if len(matches) == 1:
            return matches[0]
        else:
            return matches

    def _get_applet_by_type(self, applet_type: typing.Type[Applet]) -> typing.Union[Applet, typing.List[Applet]]:
        matches = [applet for applet in self._applets if applet.applet_type == applet_type]
        if len(matches) == 1:
            return matches[0]
        else:
            return matches


if __name__ == '__main__':
    # TODO: move to tests!
    from lazyflow.operators.opArrayPiper import OpArrayPiper
    from lazyflow.graph import Graph
    from lazyflow.operatorWrapper import OperatorWrapper

    op_pipe = OperatorWrapper(
        OpArrayPiper,
        graph=Graph()
        )

    assert op_pipe.