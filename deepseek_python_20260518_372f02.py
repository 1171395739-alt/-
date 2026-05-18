import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re
from typing import Dict, List, Optional, Tuple
import openpyxl  # for Excel support

# 页面配置
st.set_page_config(page_title="实验数据处理工具", layout="wide")

# 初始化session状态
if 'raw_files' not in st.session_state:
    st.session_state.raw_files = {}  # {文件名: DataFrame}
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'pivot_selected_index' not in st.session_state:
    st.session_state.pivot_selected_index = None

# 辅助函数: 读取文件
def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """读取CSV或Excel文件为DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"不支持的文件格式: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"读取文件 {uploaded_file.name} 失败: {str(e)}")
        return None

# 合并多文件
def merge_files(files: Dict[str, pd.DataFrame], merge_key: str, how: str = 'inner') -> pd.DataFrame:
    """合并多个DataFrame"""
    if len(files) == 0:
        return None
    dfs = list(files.values())
    if len(dfs) == 1:
        return dfs[0]
    # 逐步合并
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=merge_key, how=how, suffixes=('', '_dup'))
        # 去除重复列
        merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

# 异常标注 (基于Z-score或IQR)
def annotate_anomalies(df: pd.DataFrame, column: str, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """返回布尔Series，True表示异常"""
    data = df[column].dropna()
    if len(data) == 0:
        return pd.Series([False] * len(df), index=df.index)
    if method == 'zscore':
        zscore = np.abs((df[column] - data.mean()) / data.std())
        return zscore > threshold
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (df[column] < lower) | (df[column] > upper)
    else:
        return pd.Series([False] * len(df), index=df.index)

# 内容提取 (正则)
def extract_content(df: pd.DataFrame, source_col: str, regex: str, new_col: str) -> pd.DataFrame:
    """从指定列用正则提取内容，添加新列"""
    df[new_col] = df[source_col].astype(str).str.extract(regex, expand=False)
    return df

# 格式转换: DataFrame转Excel或CSV bytes
def to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output

def to_csv_bytes(df: pd.DataFrame) -> BytesIO:
    return BytesIO(df.to_csv(index=False).encode('utf-8'))

# 常用计算公式
def compute_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """计算均值、标准差、中位数、最大值、最小值、变异系数"""
    stats = []
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            data = df[col].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                stats.append({
                    '列名': col,
                    '均值': mean_val,
                    '标准差': std_val,
                    '中位数': data.median(),
                    '最大值': data.max(),
                    '最小值': data.min(),
                    '变异系数(CV)': std_val / mean_val if mean_val != 0 else np.nan
                })
    return pd.DataFrame(stats)

# 主UI
st.title("🔬 实验数据处理工具")
st.markdown("支持多文件合并、格式转换、异常标注、透视表与联动图表。针对实验数据设计。")

# ==================== 侧边栏: 文件管理 ====================
with st.sidebar:
    st.header("📂 数据文件管理")
    uploaded_files = st.file_uploader(
        "上传实验数据文件 (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.raw_files:
                df = load_file(file)
                if df is not None:
                    st.session_state.raw_files[file.name] = df
                    st.success(f"已加载: {file.name} (行:{df.shape[0]}, 列:{df.shape[1]})")
    
    # 显示已加载文件列表，允许删除
    if st.session_state.raw_files:
        st.subheader("已加载文件")
        for name in list(st.session_state.raw_files.keys()):
            col1, col2 = st.columns([3,1])
            col1.write(f"📄 {name}")
            if col2.button("❌", key=f"del_{name}"):
                del st.session_state.raw_files[name]
                # 重置合并数据
                st.session_state.merged_df = None
                st.session_state.processed_df = None
                st.rerun()
    else:
        st.info("暂无加载的文件，请上传。")

# ==================== 主区域: 选项卡 ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 数据整合与转换", 
    "⚠️ 异常标注 & 公式套用",
    "🔍 内容提取 & 格式转换",
    "📈 透视表 & 联动图表",
    "💾 数据导出"
])

# ---------- Tab1: 数据整合与转换(合并) ----------
with tab1:
    st.header("多文件数据合并")
    if len(st.session_state.raw_files) < 2:
        st.warning("请至少上传2个文件进行合并演示，或单文件也可直接使用。")
    else:
        st.info(f"当前已加载 {len(st.session_state.raw_files)} 个文件，请指定合并键和连接方式。")
    
    col1, col2 = st.columns(2)
    with col1:
        # 获取所有文件的列名交集作为候选合并键
        all_cols_sets = [set(df.columns) for df in st.session_state.raw_files.values()]
        common_cols = list(set.intersection(*all_cols_sets)) if all_cols_sets else []
        if common_cols:
            merge_key = st.selectbox("合并键 (各文件共有的列)", common_cols)
        else:
            merge_key = st.text_input("合并键 (手动输入，需确保所有文件均存在)", "ID")
            st.warning("未找到公共列，请手动输入一个存在于所有文件的列名。")
        
        how_merge = st.selectbox("连接方式", ['inner', 'outer', 'left', 'right'], index=0)
    
    with col2:
        if st.button("执行合并", type="primary"):
            if st.session_state.raw_files:
                merged = merge_files(st.session_state.raw_files, merge_key, how_merge)
                if merged is not None:
                    st.session_state.merged_df = merged
                    st.session_state.processed_df = merged.copy()
                    st.success(f"合并成功！数据维度: {merged.shape}")
                else:
                    st.error("合并失败，请检查合并键或数据格式。")
    
    if st.session_state.merged_df is not None:
        st.subheader("合并后数据预览 (前100行)")
        st.dataframe(st.session_state.merged_df.head(100), use_container_width=True)
        st.caption(f"总计 {st.session_state.merged_df.shape[0]} 行 x {st.session_state.merged_df.shape[1]} 列")
    else:
        if len(st.session_state.raw_files) == 1:
            # 只有一个文件时，自动设置为合并数据
            single_df = list(st.session_state.raw_files.values())[0]
            if st.session_state.merged_df is None:
                st.session_state.merged_df = single_df.copy()
                st.session_state.processed_df = single_df.copy()
                st.info("检测到单文件，已自动设置为工作数据。")
                st.dataframe(st.session_state.merged_df.head(100), use_container_width=True)

# ---------- Tab2: 异常标注 & 公式套用 ----------
with tab2:
    st.header("异常数据标注")
    if st.session_state.processed_df is not None:
        df_work = st.session_state.processed_df
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            anno_col = st.selectbox("选择要检测异常的数值列", numeric_cols, key="anno_col")
            method = st.radio("异常检测方法", ['zscore (标准分数)', 'iqr (四分位距)'], horizontal=True)
            threshold = 3.0
            if method.startswith('zscore'):
                threshold = st.slider("Z-score 阈值", 1.5, 5.0, 3.0, 0.5)
                meth = 'zscore'
            else:
                meth = 'iqr'
                threshold = 1.5  # 无需使用，仅占位
            
            if st.button("标注异常", key="run_anno"):
                anomaly_series = annotate_anomalies(df_work, anno_col, meth, threshold)
                df_work['异常标注'] = anomaly_series.map({True: '⚠️异常', False: '正常'})
                st.session_state.processed_df = df_work
                st.success(f"已在 '{anno_col}' 列上完成异常标注，新增列: '异常标注'")
                
                # 显示异常统计
                n_anomalies = anomaly_series.sum()
                st.metric("异常数据点数", n_anomalies, delta=f"{n_anomalies/len(df_work)*100:.1f}%")
            
            # 显示当前标注结果
            if '异常标注' in df_work.columns:
                st.subheader("异常标注预览 (标记为⚠️异常的行)")
                st.dataframe(df_work[df_work['异常标注'] == '⚠️异常'].head(50), use_container_width=True)
        else:
            st.warning("数据中没有数值型列，无法进行异常标注。")
        
        st.divider()
        st.header("📐 常用计算公式套用")
        if numeric_cols:
            selected_stats_cols = st.multiselect("选择要计算统计量的数值列", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            if st.button("计算统计量"):
                if selected_stats_cols:
                    stats_df = compute_stats(df_work, selected_stats_cols)
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.warning("请至少选择一列。")
        else:
            st.info("无数值列可用。")
    else:
        st.warning("请先在【数据整合与转换】中准备数据。")

# ---------- Tab3: 内容提取 & 格式转换 ----------
with tab3:
    st.header("🔎 内容提取 (正则表达式)")
    if st.session_state.processed_df is not None:
        df_extract = st.session_state.processed_df
        text_cols = df_extract.select_dtypes(include=['object', 'string']).columns.tolist()
        if text_cols:
            source_col = st.selectbox("选择源文本列", text_cols)
            regex_pattern = st.text_input("正则表达式 (例如提取数字: (\\d+\\.?\\d*))", r"(\d+\.?\d*)")
            new_col_name = st.text_input("新列名", "extracted_value")
            if st.button("执行提取"):
                df_extract = extract_content(df_extract, source_col, regex_pattern, new_col_name)
                st.session_state.processed_df = df_extract
                st.success(f"已添加新列: {new_col_name}")
                st.dataframe(df_extract[[source_col, new_col_name]].head(10), use_container_width=True)
        else:
            st.info("没有文本列，无法进行内容提取。")
    else:
        st.warning("请先准备数据。")
    
    st.divider()
    st.header("🔄 文件格式转换 (当前工作数据)")
    if st.session_state.processed_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("转换为CSV并下载"):
                csv_bytes = to_csv_bytes(st.session_state.processed_df)
                st.download_button("点击下载 CSV", data=csv_bytes, file_name="experiment_data.csv", mime="text/csv", key="csv_down")
        with col2:
            if st.button("转换为Excel并下载"):
                excel_bytes = to_excel_bytes(st.session_state.processed_df)
                st.download_button("点击下载 Excel", data=excel_bytes, file_name="experiment_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="excel_down")
    else:
        st.warning("无数据可转换。")

# ---------- Tab4: 透视表 & 联动图表 ----------
with tab4:
    st.header("数据透视表")
    if st.session_state.processed_df is not None:
        df_pivot = st.session_state.processed_df
        # 获取所有列
        all_cols = df_pivot.columns.tolist()
        numeric_cols_pivot = df_pivot.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown("### 配置透视表")
        col1, col2, col3 = st.columns(3)
        with col1:
            index_col = st.selectbox("行索引 (可多选)", all_cols, key="pivot_index", help="选择作为行标签的列")
            index_multi = st.multiselect("或多个行索引", all_cols, default=[] if not index_col else [index_col])
            if not index_multi:
                index_multi = [index_col] if index_col else None
        with col2:
            columns_col = st.selectbox("列索引", [None] + all_cols, key="pivot_columns")
        with col3:
            values_col = st.selectbox("值列 (数值)", numeric_cols_pivot if numeric_cols_pivot else all_cols, key="pivot_values")
            agg_func = st.selectbox("聚合函数", ['mean', 'sum', 'count', 'std', 'min', 'max'], index=0)
        
        if st.button("生成透视表"):
            if index_multi and values_col:
                try:
                    pivot = pd.pivot_table(df_pivot, 
                                           values=values_col, 
                                           index=index_multi, 
                                           columns=columns_col if columns_col != 'None' else None,
                                           aggfunc=agg_func,
                                           fill_value=0)
                    st.session_state['pivot_table'] = pivot
                    st.success("透视表已生成")
                except Exception as e:
                    st.error(f"透视表生成失败: {e}")
            else:
                st.warning("请选择行索引和值列")
        
        if 'pivot_table' in st.session_state:
            st.subheader("透视表结果")
            st.dataframe(st.session_state['pivot_table'], use_container_width=True)
            
            # 联动展示: 选择透视表中的某个索引值，过滤原始数据绘制趋势图
            st.divider()
            st.header("📉 联动趋势图 (基于透视表选择)")
            pivot_obj = st.session_state['pivot_table']
            # 获取行索引的可能值 (针对第一级索引)
            index_levels = pivot_obj.index.names
            if index_levels:
                selected_level = st.selectbox("选择行索引级别进行联动筛选", index_levels)
                unique_vals = pivot_obj.index.get_level_values(selected_level).unique().tolist()
                selected_val = st.selectbox(f"选择 {selected_level} 的值", unique_vals)
                # 联动: 过滤原始数据，绘制趋势图
                filtered_df = df_pivot[df_pivot[selected_level] == selected_val]
                if not filtered_df.empty:
                    # 绘制趋势图 (可选择x,y)
                    numeric_cols_plot = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols_plot) >= 1:
                        x_axis = st.selectbox("X轴 (建议选择类别/时间列)", filtered_df.columns, key="plot_x", index=0)
                        y_axis = st.selectbox("Y轴 (数值)", numeric_cols_plot, key="plot_y")
                        color_by = st.selectbox("分组着色 (可选)", [None] + filtered_df.columns.tolist(), key="plot_color")
                        if st.button("绘制联动趋势图"):
                            fig = px.line(filtered_df, x=x_axis, y=y_axis, color=color_by, 
                                          title=f"数据趋势 (筛选条件: {selected_level}={selected_val})",
                                          markers=True)
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 也可绘制散点图/柱状图，提供切换
                            chart_type = st.radio("图表类型", ["折线图", "散点图", "柱状图"], horizontal=True)
                            if chart_type == "散点图":
                                fig2 = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by, title="散点图联动")
                                st.plotly_chart(fig2, use_container_width=True)
                            elif chart_type == "柱状图":
                                fig3 = px.bar(filtered_df, x=x_axis, y=y_axis, color=color_by, title="柱状图联动")
                                st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("过滤后的数据没有数值列，无法绘图")
                else:
                    st.warning("没有符合条件的数据")
            else:
                st.info("透视表索引结构简单，直接基于当前数据绘制全局趋势图（无联动）")
                # 提供一个全数据趋势图选择
                st.subheader("全数据趋势图")
                numeric_cols_plot = df_pivot.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols_plot:
                    x_axis_all = st.selectbox("X轴", df_pivot.columns, key="x_all")
                    y_axis_all = st.selectbox("Y轴", numeric_cols_plot, key="y_all")
                    color_all = st.selectbox("分组", [None] + df_pivot.columns.tolist(), key="color_all")
                    fig_all = px.line(df_pivot, x=x_axis_all, y=y_axis_all, color=color_all, markers=True, title="全局趋势")
                    st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.warning("请先在数据整合中准备数据。")

# ---------- Tab5: 数据导出 ----------
with tab5:
    st.header("💾 导出最终处理后的数据")
    if st.session_state.processed_df is not None:
        st.subheader("当前工作数据 (包含异常标注、内容提取等)")
        st.dataframe(st.session_state.processed_df.head(50), use_container_width=True)
        csv_final = to_csv_bytes(st.session_state.processed_df)
        excel_final = to_excel_bytes(st.session_state.processed_df)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 导出为 CSV", data=csv_final, file_name="processed_experiment_data.csv", mime="text/csv")
        with col2:
            st.download_button("📥 导出为 Excel", data=excel_final, file_name="processed_experiment_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("注: 导出的数据包含所有处理步骤结果。")
    else:
        st.warning("无数据可导出。")

st.sidebar.markdown("---")
st.sidebar.info(
    "**使用说明**\n"
    "1. 左侧上传实验数据文件(CSV/Excel)，支持多文件。\n"
    "2. 在【数据整合】合并多个文件(可选)。\n"
    "3. 【异常标注】基于Z-score/IQR标记异常点，新增标注列。\n"
    "4. 【公式套用】计算常用统计量。\n"
    "5. 【内容提取】用正则从文本列提取信息。\n"
    "6. 【格式转换】可将当前数据下载为CSV/Excel。\n"
    "7. 【透视表&联动】生成透视表，并单击联动筛选原始数据绘制趋势图。\n"
    "8. 所有处理结果保存在会话中，最终可在【数据导出】下载。"
)