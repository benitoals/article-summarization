--- Debug Example 0 ---
Input (first 300 chars): Empirical Study of PLC Authentication Protocols in Industrial Control Systems Adeen Ayub Department of Computer Science Virginia Commonwealth University Richmond, United States of America ayuba2@vcu.eduHyunguk Yoo Department of Computer Science The University of New Orleans New Orleans, United State
Predicted Summary: <extra_id_0>-logic systems (PLCs) and PLCs. These systems are controlled by a PLC. The PLC uses a control-logic program (PLC) and the PLC (PLC). In industrial control systems (ICS), the ICS uses an authentication protocol that allows PLC to control the physical process. However, most PLC systems use a software-based software to control physical processes. The software uses the control logic of PLC, PLC is based on the remote control systems. This paper presents an empirical study on
Reference Summary: Programmable logic controllers (PLCs) run a  con- trol logic  program that de nes how to control a physicalprocess such as a nuclear plant, power grid stations, and gas pipelines. Attackers target the control logic of a PLC to sabotage a physical process. Most PLCs employ password- based authentication mechanisms to prevent unauthorized remoteaccess to control logic. This paper presents an empirical study on proprietary authentication mechanisms in  ve industry-scale PLCs to understand the security-design practices of four popular ICS vendors, i.e., Allen-Bradley, Schneider Electric, Automa-tionDirect, and Siemens. The empirical study determines whether the mechanisms are vulnerable by design and can be exploited. It reveals serious design issues and vulnerabilities in authentication mechanisms, including lack of nonce, small-sized encryptionkey, weak encryption scheme, and client-side authentication. The study further con rms the  ndings empirically by creatingand testing their proof-of-concept exploits derived from MITRE ATT&CK knowledge base of adversary tactics and techniques. Unlike existing work, our study relies solely on network traf c examination and does not employ typical reverse-engineeringof binary  les (e.g., PLC  rmware) to reveal the seriousness of design problems. Moreover, the study covers PLCs from different vendors to highlight an industry-wide issue of secure PLC authentication that needs to be addressed.
--------------------------------------------------

--- Debug Example 1 ---
Input (first 300 chars): On Experimental validation of Whitelist Auto- Generation Method for Secured Programmable Logic Controllers Shintaro Fujita Dept. of Mechanical Engineering and Intelligent Systems University of Electro-Communications Tokyo, Japan shntr.fujita@uec.ac.jp Kenji Sawada Info-Powerd Energy System Research 
Predicted Summary: <extra_id_0>, PLC and Intelligent Systems. The typical control system is controlled by SCADAs. However, there is a few attacks. In this context, we need to avoid attacks on PLCs. It is very difficult to detect SCADA systems. The most important attacks are known as SCADA attacks which are being attacked by PLC systems. This is critical to prevent attacks from SCADA. The main attacks of SCADA is called SCADA (SCADA) and SCADA, which is responsible for PLC
Reference Summary: This paper considers a whitelisting system for programmable logic controllers (PLCs). In control systems, controllers are final fortresses to continues the operation of field devices (actuators/sensors), but they are fragile with respect to malware and zero-day attacks. One of the countermeasures applicable for controllers is a whitelisting system which registers normal operations of controller behavior in a  whitelist  to detect abnormal operations via a whitelist. The previous research of the current author proposed a PLC whitelisting system with a control via a ladder diagram (LD). LD representations have a wide applicability because LDs can be implemented for all PLCs and security functions without hardware/firmware updates. However, the current status requires that all instances are manually entered in the whitelist. In this talk, we show how the setting up of the can be automatized whitelist from the PLC behavior. This paper introduces an auto-generation approach for the whitelist using sequential function chart (SFC) instead of the LD. SFC and LD are compatible representations for the PLC. Using Petri Net modeling, this paper proposes how to generate the whitelist from the SFC and how to detect abnormal operations via the whitelist. We call the SFC-based approach the model-based whitelist, the Petri Net based approach the model-based detection. Further, this paper carries out an experimental validation of the algorithms using an OpenPLC based testbed system.
--------------------------------------------------

--- Debug Example 2 ---
Input (first 300 chars): IEEE INTERNET OF THINGS JOURNAL, VOL. 9, NO. 15, 1 AUGUST 2022 13223 Moving Target Defense for Cyber Physical Systems Using IoT-Enabled Data Replication Jairo A. Giraldo ,Member, IEEE , Mohamad El Hariri ,Member, IEEE , and Masood Parvania ,Senior Member, IEEE Index Terms  Cyber physical systems, cy
Predicted Summary: <extra_id_0> physical systems (CPSs) is a physical system that could be controlled by software. In this context, there are significant vulnerability attacks that have been threatened by critical infrastructure systems, such as infrastructure and infrastructure, which are known as CPSs, including physical infrastructure attacks. This work is supported by the critical systems and systems, as well as the IoT attacks, using IoT-enabled data replication. This paper is focused on the physical environment. In addition, there is significantly
Reference Summary: This article proposes a novel moving target defense (MTD) strategy that leverages the versatility of the Internetof Things (IoT) networks to enhance the security of cyber physical systems (CPSs) by replicating relevant sensory andcontrol signals. The replicated data are randomly selected andtransmitted to create two layers of uncertainties that reduce theability of adversaries to launch successful cyberattacks, withoutaffecting the performance of the system in a normal operation.The theoretical foundations of designing the IoT network andoptimal allocation of replicas per signal are developed for linear-time-invariant systems, and fundamental limits of uncertaintiesintroduced by the framework are calculated. The orchestration of the layers and applications integrated in the proposed framework is demonstrated in experimental implementation on a real-timewater system over a WiFi network, adopting a data-centricarchitecture. The implementation results demonstrate that theproposed framework considerably limits the impact of false-data-injection attacks, while decreasing the ability of adversaries tolearn details about the physical system operation.
--------------------------------------------------

=== Final Model (HF + Local) ===
{'rouge1': np.float64(23.756683347976633), 'rouge2': np.float64(4.026777692110272), 'rougeL': np.float64(13.063501783652884), 'rougeLsum': np.float64(13.181579754281355)}

===== All Four Evaluation Results =====
1) Baseline (Pretrained Model)      => {'rouge1': np.float64(0.4076244552579688), 'rouge2': np.float64(0.12202562538133008), 'rougeL': np.float64(0.4076244552579688), 'rougeLsum': np.float64(0.4076244552579688)}
2) LoRA on Local Dataset            => {'rouge1': np.float64(21.31466789375662), 'rouge2': np.float64(4.293126888205344), 'rougeL': np.float64(11.594329113813176), 'rougeLsum': np.float64(11.558987333935299)}
3) LoRA on HF Dataset               => {'rouge1': np.float64(19.881724986527278), 'rouge2': np.float64(2.5562797012453835), 'rougeL': np.float64(13.16161974619779), 'rougeLsum': np.float64(13.298388820716106)}
4) Fine-tuned HF + Local Model      => {'rouge1': np.float64(23.756683347976633), 'rouge2': np.float64(4.026777692110272), 'rougeL': np.float64(13.063501783652884), 'rougeLsum': np.float64(13.181579754281355)}


Below is a table summarizing and comparing the evaluation results for each experiment:

Model	ROUGE-1	ROUGE-2	ROUGE-L	ROUGE-Lsum
Baseline (Pretrained Model)	0.41	0.12	0.41	0.41
LoRA on Local Dataset	21.31	4.29	11.59	11.56
LoRA on HF Dataset	19.88	2.56	13.16	13.30
Fine-tuned HF + Local Model	23.76	4.03	13.06	13.18

Note: The values are rounded to two decimal places for clarity.
This table shows that the pretrained baseline performs very poorly compared to fine‑tuned models, with the best performance achieved by the combined fine‑tuning (HF + Local).

Here’s an explanation of the results:
	1.	Baseline (Pretrained Model):
	•	ROUGE Scores: The baseline shows very low ROUGE values (around 0.41 for ROUGE-1 and ROUGE-L, and 0.12 for ROUGE-2).
	•	Interpretation:
	•	This indicates that without any fine‑tuning, the pretrained model generates summaries that barely overlap with the reference summaries.
	•	The model, as pretrained, isn’t well-adapted for your specific summarization task or domain.
	2.	LoRA on Local Dataset:
	•	ROUGE Scores: ROUGE-1 jumps to 21.31, ROUGE-2 to 4.29, and ROUGE-L / ROUGE-Lsum to around 11.59.
	•	Interpretation:
	•	Fine‑tuning the model using a LoRA adapter on your local dataset significantly improves performance.
	•	The improvement suggests that even a relatively small local dataset can effectively adjust the model’s summarization capabilities for your domain.
	3.	LoRA on HF Dataset:
	•	ROUGE Scores: ROUGE-1 is 19.88, ROUGE-2 drops to 2.56, but ROUGE-L and ROUGE-Lsum increase to around 13.16 and 13.30 respectively.
	•	Interpretation:
	•	Training on the larger HF dataset (which is likely from a different domain) results in decent overall overlap in longer spans (as indicated by ROUGE-L and ROUGE-Lsum), but the lower ROUGE-2 suggests that the precision of 2-gram overlaps is reduced.
	•	This outcome may be due to a domain mismatch—the HF dataset may not perfectly align with your local dataset’s style or content.
	4.	Fine-tuned HF + Local Model:
	•	ROUGE Scores: The combined fine‑tuning achieves the best ROUGE-1 (23.76) and competitive ROUGE-2 (4.03) and ROUGE-L/ROUGE-Lsum (around 13.06–13.18).
	•	Interpretation:
	•	This final stage uses a model initially fine‑tuned on the HF dataset and then further fine‑tuned on your local data (with the base frozen so that only the adapter is updated).
	•	The combination allows the model to leverage the general patterns learned from the larger HF dataset while adapting to the specific characteristics of your local domain.
	•	The result is the best overall performance among the tested configurations, as measured by ROUGE-1 and good performance on ROUGE-L and ROUGE-Lsum, indicating improved content and fluency in the summaries.

Overall Conclusions
	•	Importance of Fine‑Tuning:
The very low scores of the baseline model highlight that a pretrained model must be fine‑tuned for effective summarization on your specific dataset.
	•	Domain Adaptation Matters:
Fine‑tuning on your local dataset yields significant improvements, but combining training on a larger (HF) dataset followed by fine‑tuning on local data provides the best performance. This suggests that leveraging broader knowledge from the HF dataset, even if it’s slightly mismatched, can be beneficial when later refined with domain-specific data.
	•	Metric Differences:
	•	ROUGE-1 (unigram overlap) increases the most in the combined model, indicating better overall content match.
	•	ROUGE-2 (bigram overlap) is relatively low in the HF-only model but improves when combined with local data.
	•	ROUGE-L/ROUGE-Lsum (longest common subsequence) being higher in the HF and combined models suggests that the generated summaries capture longer, more coherent segments of the reference summaries.

These results collectively indicate that a two-stage fine‑tuning process (first on a large, general dataset, then on a small, local dataset with only the adapter updated) yields the best summarization performance for your task.
