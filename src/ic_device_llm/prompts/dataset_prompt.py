from pydantic import BaseModel

SYSTEM_PROMPT = "You are a helpful assistant that can infer the ids of a GAA based on the input information."


class UserPrompt(BaseModel):
    lg_nw: float
    r_nw: float
    t_ox_nm: float
    vg: float
    vd: float
    base_prompt: str = """\
    GAA device parameters:
        - Gate length (lg_nw): {lg_nw} nm
        - Nanowire radius (r_nw): {r_nw} nm
        - Oxide thickness (t_ox_nm): {t_ox_nm} nm
        - Gate voltage (vg): {vg} V
        - Drain voltage (vd): {vd} V
    Infer the drain saturation current (ids) for this GAA device."""
    # ids
    # base_prompt: str = """\
    # This GAA (Gate-All-Around) device has the following properties:
    #
    # 1. lg_nw (Gate Length Nanowire): {lg_nw} nm
    #    This is the length of the gate along the nanowire direction, affecting the device's switching characteristics and short channel effects.
    #
    # 2. r_nw (Radius Nanowire): {r_nw} nm
    #    This is the radius of the nanowire forming the transistor channel, influencing current drive capability and quantum effects.
    #
    # 3. t_ox_nm (Oxide Thickness): {t_ox_nm} nm
    #    This is the thickness of the gate oxide layer surrounding the nanowire, impacting gate capacitance and control capability.
    #
    # 4. vg (Gate Voltage): {vg} V
    #    This is the voltage applied to the gate, used to control the conduction state of the channel.
    #
    # 5. vd (Drain Voltage): {vd} V
    #    This is the voltage applied to the drain, affecting drain current and the device's saturation characteristics.
    #
    # Based on these parameters, please infer the ids (drain saturation current) of this GAA device.
    # """

    def render_prompt(self) -> str:
        return self.base_prompt.format(
            lg_nw=self.lg_nw,
            r_nw=self.r_nw,
            t_ox_nm=self.t_ox_nm,
            vg=self.vg,
            vd=self.vd,
        )


class AssistantPrompt(BaseModel):
    ids: float
    base_prompt: str = """The ids of this device is {ids} A."""

    def render_prompt(self) -> str:
        return self.base_prompt.format(ids=self.ids)
