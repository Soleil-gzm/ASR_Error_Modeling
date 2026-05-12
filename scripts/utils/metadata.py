from pathlib import Path

def get_step_output(metadata, step_name, key='output_csv', default=None):
    """从元数据中获取指定步骤的输出文件路径（适配新旧格式）"""
    step_info = metadata.get(step_name, {})
    if key in step_info:
        return step_info[key]
    if 'latest' in step_info and key in step_info['latest']:
        return step_info['latest'][key]
    return default

def get_step_sample_ratio(metadata, step_name='01_compute_sentence_nll'):
    """获取采样比例（适配新旧格式）"""
    step_info = metadata.get(step_name, {})
    if 'sample_ratio' in step_info:
        return step_info['sample_ratio']
    if 'latest' in step_info and 'sample_ratio' in step_info['latest']:
        return step_info['latest']['sample_ratio']
    return 1.0