# THIS FILE WAS AUTOMATICALLY GENERATED AT 2022-02-01T22:18:41.894374
#####################################################################

from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class SynthParams(SQLModel):
	id: Optional[int] = Field(primary_key=True, default=None)
	MasterVol: float = Field(alias="0")
	A_Vol: float = Field(alias="1")
	A_Pan: float = Field(alias="2")
	A_Octave: float = Field(alias="3")
	A_Semi: float = Field(alias="4")
	A_Fine: float = Field(alias="5")
	A_Unison: float = Field(alias="6")
	A_UniDet: float = Field(alias="7")
	A_UniBlend: float = Field(alias="8")
	A_Warp: float = Field(alias="9")
	A_CoarsePit: float = Field(alias="10")
	A_WTPos: float = Field(alias="11")
	A_RandPhase: float = Field(alias="12")
	A_Phase: float = Field(alias="13")
	B_Vol: float = Field(alias="14")
	B_Pan: float = Field(alias="15")
	B_Octave: float = Field(alias="16")
	B_Semi: float = Field(alias="17")
	B_Fine: float = Field(alias="18")
	B_Unison: float = Field(alias="19")
	B_UniDet: float = Field(alias="20")
	B_UniBlend: float = Field(alias="21")
	B_Warp: float = Field(alias="22")
	B_CoarsePit: float = Field(alias="23")
	B_WTPos: float = Field(alias="24")
	B_RandPhase: float = Field(alias="25")
	B_Phase: float = Field(alias="26")
	Noise_Level: float = Field(alias="27")
	Noise_Pitch: float = Field(alias="28")
	Noise_Fine: float = Field(alias="29")
	Noise_Pan: float = Field(alias="30")
	Noise_RandPhase: float = Field(alias="31")
	Noise_Phase: float = Field(alias="32")
	Sub_Osc_Level: float = Field(alias="33")
	Sub_Osc_Pan: float = Field(alias="34")
	Env1_Atk: float = Field(alias="35")
	Env1_Hold: float = Field(alias="36")
	Env1_Dec: float = Field(alias="37")
	Env1_Sus: float = Field(alias="38")
	Env1_Rel: float = Field(alias="39")
	OscA_gt_Fil: float = Field(alias="40")
	OscB_gt_Fil: float = Field(alias="41")
	OscN_gt_Fil: float = Field(alias="42")
	OscS_gt_Fil: float = Field(alias="43")
	Fil_Type: float = Field(alias="44")
	Fil_Cutoff: float = Field(alias="45")
	Fil_Reso: float = Field(alias="46")
	Fil_Driv: float = Field(alias="47")
	Fil_Var: float = Field(alias="48")
	Fil_Mix: float = Field(alias="49")
	Fil_Stereo: float = Field(alias="50")
	Env2_Atk: float = Field(alias="51")
	Env2_Hld: float = Field(alias="52")
	Env2_Dec: float = Field(alias="53")
	Env2_Sus: float = Field(alias="54")
	Env2_Rel: float = Field(alias="55")
	Env3_Atk: float = Field(alias="56")
	Env3_Hld: float = Field(alias="57")
	Env3_Dec: float = Field(alias="58")
	Env3_Sus: float = Field(alias="59")
	Env3_Rel: float = Field(alias="60")
	LFO1_Rate: float = Field(alias="61")
	LFO2_Rate: float = Field(alias="62")
	LFO3_Rate: float = Field(alias="63")
	LFO4_Rate: float = Field(alias="64")
	PortTime: float = Field(alias="65")
	PortCurve: float = Field(alias="66")
	Chaos1_BPM: float = Field(alias="67")
	Chaos2_BPM: float = Field(alias="68")
	Chaos1_Rate: float = Field(alias="69")
	Chaos2_Rate: float = Field(alias="70")
	A_curve1: float = Field(alias="71")
	D_curve1: float = Field(alias="72")
	R_curve1: float = Field(alias="73")
	A_curve2: float = Field(alias="74")
	D_curve2: float = Field(alias="75")
	R_curve2: float = Field(alias="76")
	A_curve3: float = Field(alias="77")
	D_curve3: float = Field(alias="78")
	R_curve3: float = Field(alias="79")
	Mast_dot_Tun: float = Field(alias="80")
	Verb_Wet: float = Field(alias="81")
	VerbSize: float = Field(alias="82")
	Decay: float = Field(alias="83")
	VerbLoCt: float = Field(alias="84")
	Spin_Rate: float = Field(alias="85")
	VerbHiCt: float = Field(alias="86")
	Spin_Depth: float = Field(alias="87")
	EQ_FrqL: float = Field(alias="88")
	EQ_FrqH: float = Field(alias="89")
	EQ_Q_L: float = Field(alias="90")
	EQ_Q_H: float = Field(alias="91")
	EQ_VolL: float = Field(alias="92")
	EQ_VolH: float = Field(alias="93")
	EQ_TypL: float = Field(alias="94")
	EQ_TypH: float = Field(alias="95")
	Dist_Wet: float = Field(alias="96")
	Dist_Drv: float = Field(alias="97")
	Dist_L_slash_B_slash_H: float = Field(alias="98")
	Dist_Mode: float = Field(alias="99")
	Dist_Freq: float = Field(alias="100")
	Dist_BW: float = Field(alias="101")
	Dist_PrePost: float = Field(alias="102")
	Flg_Wet: float = Field(alias="103")
	Flg_BPM_Sync: float = Field(alias="104")
	Flg_Rate: float = Field(alias="105")
	Flg_Dep: float = Field(alias="106")
	Flg_Feed: float = Field(alias="107")
	Flg_Stereo: float = Field(alias="108")
	Phs_Wet: float = Field(alias="109")
	Phs_BPM_Sync: float = Field(alias="110")
	Phs_Rate: float = Field(alias="111")
	Phs_Dpth: float = Field(alias="112")
	Phs_Frq: float = Field(alias="113")
	Phs_Feed: float = Field(alias="114")
	Phs_Stereo: float = Field(alias="115")
	Cho_Wet: float = Field(alias="116")
	Cho_BPM_Sync: float = Field(alias="117")
	Cho_Rate: float = Field(alias="118")
	Cho_Dly: float = Field(alias="119")
	Cho_Dly2: float = Field(alias="120")
	Cho_Dep: float = Field(alias="121")
	Cho_Feed: float = Field(alias="122")
	Cho_Filt: float = Field(alias="123")
	Dly_Wet: float = Field(alias="124")
	Dly_Freq: float = Field(alias="125")
	Dly_BW: float = Field(alias="126")
	Dly_BPM_Sync: float = Field(alias="127")
	Dly_Link: float = Field(alias="128")
	Dly_TimL: float = Field(alias="129")
	Dly_TimR: float = Field(alias="130")
	Dly_Mode: float = Field(alias="131")
	Dly_Feed: float = Field(alias="132")
	Dly_Off_L: float = Field(alias="133")
	Dly_Off_R: float = Field(alias="134")
	Cmp_Thr: float = Field(alias="135")
	Cmp_Rat: float = Field(alias="136")
	Cmp_Att: float = Field(alias="137")
	Cmp_Rel: float = Field(alias="138")
	CmpGain: float = Field(alias="139")
	CmpMBnd: float = Field(alias="140")
	FX_Fil_Wet: float = Field(alias="141")
	FX_Fil_Type: float = Field(alias="142")
	FX_Fil_Freq: float = Field(alias="143")
	FX_Fil_Reso: float = Field(alias="144")
	FX_Fil_Drive: float = Field(alias="145")
	FX_Fil_Var: float = Field(alias="146")
	Hyp_Wet: float = Field(alias="147")
	Hyp_Rate: float = Field(alias="148")
	Hyp_Detune: float = Field(alias="149")
	Hyp_Unison: float = Field(alias="150")
	Hyp_Retrig: float = Field(alias="151")
	HypDim_Size: float = Field(alias="152")
	HypDim_Mix: float = Field(alias="153")
	Dist_Enable: float = Field(alias="154")
	Flg_Enable: float = Field(alias="155")
	Phs_Enable: float = Field(alias="156")
	Cho_Enable: float = Field(alias="157")
	Dly_Enable: float = Field(alias="158")
	Comp_Enable: float = Field(alias="159")
	Rev_Enable: float = Field(alias="160")
	EQ_Enable: float = Field(alias="161")
	FX_Fil_Enable: float = Field(alias="162")
	Hyp_Enable: float = Field(alias="163")
	OscAPitchTrack: float = Field(alias="164")
	OscBPitchTrack: float = Field(alias="165")
	Bend_U: float = Field(alias="166")
	Bend_D: float = Field(alias="167")
	WarpOscA: float = Field(alias="168")
	WarpOscB: float = Field(alias="169")
	SubOscShape: float = Field(alias="170")
	SubOscOctave: float = Field(alias="171")
	A_Uni_LR: float = Field(alias="172")
	B_Uni_LR: float = Field(alias="173")
	A_Uni_Warp: float = Field(alias="174")
	B_Uni_Warp: float = Field(alias="175")
	A_Uni_WTPos: float = Field(alias="176")
	B_Uni_WTPos: float = Field(alias="177")
	A_Uni_Stack: float = Field(alias="178")
	B_Uni_Stack: float = Field(alias="179")
	Mod_1_amt: float = Field(alias="180")
	Mod_1_out: float = Field(alias="181")
	Mod_2_amt: float = Field(alias="182")
	Mod_2_out: float = Field(alias="183")
	Mod_3_amt: float = Field(alias="184")
	Mod_3_out: float = Field(alias="185")
	Mod_4_amt: float = Field(alias="186")
	Mod_4_out: float = Field(alias="187")
	Mod_5_amt: float = Field(alias="188")
	Mod_5_out: float = Field(alias="189")
	Mod_6_amt: float = Field(alias="190")
	Mod_6_out: float = Field(alias="191")
	Mod_7_amt: float = Field(alias="192")
	Mod_7_out: float = Field(alias="193")
	Mod_8_amt: float = Field(alias="194")
	Mod_8_out: float = Field(alias="195")
	Mod_9_amt: float = Field(alias="196")
	Mod_9_out: float = Field(alias="197")
	Mod10_amt: float = Field(alias="198")
	Mod10_out: float = Field(alias="199")
	Mod11_amt: float = Field(alias="200")
	Mod11_out: float = Field(alias="201")
	Mod12_amt: float = Field(alias="202")
	Mod12_out: float = Field(alias="203")
	Mod13_amt: float = Field(alias="204")
	Mod13_out: float = Field(alias="205")
	Mod14_amt: float = Field(alias="206")
	Mod14_out: float = Field(alias="207")
	Mod15_amt: float = Field(alias="208")
	Mod15_out: float = Field(alias="209")
	Mod16_amt: float = Field(alias="210")
	Mod16_out: float = Field(alias="211")
	Osc_A_On: float = Field(alias="212")
	Osc_B_On: float = Field(alias="213")
	Osc_N_On: float = Field(alias="214")
	Osc_S_On: float = Field(alias="215")
	Filter_On: float = Field(alias="216")
	Mod_Wheel: float = Field(alias="217")
	Macro_1: float = Field(alias="218")
	Macro_2: float = Field(alias="219")
	Macro_3: float = Field(alias="220")
	Macro_4: float = Field(alias="221")
	Amp_dot_: float = Field(alias="222")
	LFO1_smooth: float = Field(alias="223")
	LFO2_smooth: float = Field(alias="224")
	LFO3_smooth: float = Field(alias="225")
	LFO4_smooth: float = Field(alias="226")
	Pitch_Bend: float = Field(alias="227")
	Mod17_amt: float = Field(alias="228")
	Mod17_out: float = Field(alias="229")
	Mod18_amt: float = Field(alias="230")
	Mod18_out: float = Field(alias="231")
	Mod19_amt: float = Field(alias="232")
	Mod19_out: float = Field(alias="233")
	Mod20_amt: float = Field(alias="234")
	Mod20_out: float = Field(alias="235")
	Mod21_amt: float = Field(alias="236")
	Mod21_out: float = Field(alias="237")
	Mod22_amt: float = Field(alias="238")
	Mod22_out: float = Field(alias="239")
	Mod23_amt: float = Field(alias="240")
	Mod23_out: float = Field(alias="241")
	Mod24_amt: float = Field(alias="242")
	Mod24_out: float = Field(alias="243")
	Mod25_amt: float = Field(alias="244")
	Mod25_out: float = Field(alias="245")
	Mod26_amt: float = Field(alias="246")
	Mod26_out: float = Field(alias="247")
	Mod27_amt: float = Field(alias="248")
	Mod27_out: float = Field(alias="249")
	Mod28_amt: float = Field(alias="250")
	Mod28_out: float = Field(alias="251")
	Mod29_amt: float = Field(alias="252")
	Mod29_out: float = Field(alias="253")
	Mod30_amt: float = Field(alias="254")
	Mod30_out: float = Field(alias="255")
	Mod31_amt: float = Field(alias="256")
	Mod31_out: float = Field(alias="257")
	Mod32_amt: float = Field(alias="258")
	Mod32_out: float = Field(alias="259")
	LFO5_Rate: float = Field(alias="260")
	LFO6_Rate: float = Field(alias="261")
	LFO7_Rate: float = Field(alias="262")
	LFO8_Rate: float = Field(alias="263")
	LFO5_smooth: float = Field(alias="264")
	LFO6_smooth: float = Field(alias="265")
	LFO7_smooth: float = Field(alias="266")
	LFO8_smooth: float = Field(alias="267")
	FX_Fil_Pan: float = Field(alias="268")
	Comp_Wet: float = Field(alias="269")
	CompMB_L: float = Field(alias="270")
	CompMB_M: float = Field(alias="271")
	CompMB_H: float = Field(alias="272")

class SynthParamsTable(SynthParams, table=True):
	__tablename__ = "SynthParams"
	__table_args__ = (UniqueConstraint(
		"MasterVol",
		"A_Vol",
		"A_Pan",
		"A_Octave",
		"A_Semi",
		"A_Fine",
		"A_Unison",
		"A_UniDet",
		"A_UniBlend",
		"A_Warp",
		"A_CoarsePit",
		"A_WTPos",
		"A_RandPhase",
		"A_Phase",
		"B_Vol",
		"B_Pan",
		"B_Octave",
		"B_Semi",
		"B_Fine",
		"B_Unison",
		"B_UniDet",
		"B_UniBlend",
		"B_Warp",
		"B_CoarsePit",
		"B_WTPos",
		"B_RandPhase",
		"B_Phase",
		"Noise_Level",
		"Noise_Pitch",
		"Noise_Fine",
		"Noise_Pan",
		"Noise_RandPhase",
		"Noise_Phase",
		"Sub_Osc_Level",
		"Sub_Osc_Pan",
		"Env1_Atk",
		"Env1_Hold",
		"Env1_Dec",
		"Env1_Sus",
		"Env1_Rel",
		"OscA_gt_Fil",
		"OscB_gt_Fil",
		"OscN_gt_Fil",
		"OscS_gt_Fil",
		"Fil_Type",
		"Fil_Cutoff",
		"Fil_Reso",
		"Fil_Driv",
		"Fil_Var",
		"Fil_Mix",
		"Fil_Stereo",
		"Env2_Atk",
		"Env2_Hld",
		"Env2_Dec",
		"Env2_Sus",
		"Env2_Rel",
		"Env3_Atk",
		"Env3_Hld",
		"Env3_Dec",
		"Env3_Sus",
		"Env3_Rel",
		"LFO1_Rate",
		"LFO2_Rate",
		"LFO3_Rate",
		"LFO4_Rate",
		"PortTime",
		"PortCurve",
		"Chaos1_BPM",
		"Chaos2_BPM",
		"Chaos1_Rate",
		"Chaos2_Rate",
		"A_curve1",
		"D_curve1",
		"R_curve1",
		"A_curve2",
		"D_curve2",
		"R_curve2",
		"A_curve3",
		"D_curve3",
		"R_curve3",
		"Mast_dot_Tun",
		"Verb_Wet",
		"VerbSize",
		"Decay",
		"VerbLoCt",
		"Spin_Rate",
		"VerbHiCt",
		"Spin_Depth",
		"EQ_FrqL",
		"EQ_FrqH",
		"EQ_Q_L",
		"EQ_Q_H",
		"EQ_VolL",
		"EQ_VolH",
		"EQ_TypL",
		"EQ_TypH",
		"Dist_Wet",
		"Dist_Drv",
		"Dist_L_slash_B_slash_H",
		"Dist_Mode",
		"Dist_Freq",
		"Dist_BW",
		"Dist_PrePost",
		"Flg_Wet",
		"Flg_BPM_Sync",
		"Flg_Rate",
		"Flg_Dep",
		"Flg_Feed",
		"Flg_Stereo",
		"Phs_Wet",
		"Phs_BPM_Sync",
		"Phs_Rate",
		"Phs_Dpth",
		"Phs_Frq",
		"Phs_Feed",
		"Phs_Stereo",
		"Cho_Wet",
		"Cho_BPM_Sync",
		"Cho_Rate",
		"Cho_Dly",
		"Cho_Dly2",
		"Cho_Dep",
		"Cho_Feed",
		"Cho_Filt",
		"Dly_Wet",
		"Dly_Freq",
		"Dly_BW",
		"Dly_BPM_Sync",
		"Dly_Link",
		"Dly_TimL",
		"Dly_TimR",
		"Dly_Mode",
		"Dly_Feed",
		"Dly_Off_L",
		"Dly_Off_R",
		"Cmp_Thr",
		"Cmp_Rat",
		"Cmp_Att",
		"Cmp_Rel",
		"CmpGain",
		"CmpMBnd",
		"FX_Fil_Wet",
		"FX_Fil_Type",
		"FX_Fil_Freq",
		"FX_Fil_Reso",
		"FX_Fil_Drive",
		"FX_Fil_Var",
		"Hyp_Wet",
		"Hyp_Rate",
		"Hyp_Detune",
		"Hyp_Unison",
		"Hyp_Retrig",
		"HypDim_Size",
		"HypDim_Mix",
		"Dist_Enable",
		"Flg_Enable",
		"Phs_Enable",
		"Cho_Enable",
		"Dly_Enable",
		"Comp_Enable",
		"Rev_Enable",
		"EQ_Enable",
		"FX_Fil_Enable",
		"Hyp_Enable",
		"OscAPitchTrack",
		"OscBPitchTrack",
		"Bend_U",
		"Bend_D",
		"WarpOscA",
		"WarpOscB",
		"SubOscShape",
		"SubOscOctave",
		"A_Uni_LR",
		"B_Uni_LR",
		"A_Uni_Warp",
		"B_Uni_Warp",
		"A_Uni_WTPos",
		"B_Uni_WTPos",
		"A_Uni_Stack",
		"B_Uni_Stack",
		"Mod_1_amt",
		"Mod_1_out",
		"Mod_2_amt",
		"Mod_2_out",
		"Mod_3_amt",
		"Mod_3_out",
		"Mod_4_amt",
		"Mod_4_out",
		"Mod_5_amt",
		"Mod_5_out",
		"Mod_6_amt",
		"Mod_6_out",
		"Mod_7_amt",
		"Mod_7_out",
		"Mod_8_amt",
		"Mod_8_out",
		"Mod_9_amt",
		"Mod_9_out",
		"Mod10_amt",
		"Mod10_out",
		"Mod11_amt",
		"Mod11_out",
		"Mod12_amt",
		"Mod12_out",
		"Mod13_amt",
		"Mod13_out",
		"Mod14_amt",
		"Mod14_out",
		"Mod15_amt",
		"Mod15_out",
		"Mod16_amt",
		"Mod16_out",
		"Osc_A_On",
		"Osc_B_On",
		"Osc_N_On",
		"Osc_S_On",
		"Filter_On",
		"Mod_Wheel",
		"Macro_1",
		"Macro_2",
		"Macro_3",
		"Macro_4",
		"Amp_dot_",
		"LFO1_smooth",
		"LFO2_smooth",
		"LFO3_smooth",
		"LFO4_smooth",
		"Pitch_Bend",
		"Mod17_amt",
		"Mod17_out",
		"Mod18_amt",
		"Mod18_out",
		"Mod19_amt",
		"Mod19_out",
		"Mod20_amt",
		"Mod20_out",
		"Mod21_amt",
		"Mod21_out",
		"Mod22_amt",
		"Mod22_out",
		"Mod23_amt",
		"Mod23_out",
		"Mod24_amt",
		"Mod24_out",
		"Mod25_amt",
		"Mod25_out",
		"Mod26_amt",
		"Mod26_out",
		"Mod27_amt",
		"Mod27_out",
		"Mod28_amt",
		"Mod28_out",
		"Mod29_amt",
		"Mod29_out",
		"Mod30_amt",
		"Mod30_out",
		"Mod31_amt",
		"Mod31_out",
		"Mod32_amt",
		"Mod32_out",
		"LFO5_Rate",
		"LFO6_Rate",
		"LFO7_Rate",
		"LFO8_Rate",
		"LFO5_smooth",
		"LFO6_smooth",
		"LFO7_smooth",
		"LFO8_smooth",
		"FX_Fil_Pan",
		"Comp_Wet",
		"CompMB_L",
		"CompMB_M",
		"CompMB_H"
	),)
