$(document).ready(function() {
    // 处理表单提交
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();
        
        var formData = new FormData(this);
        var fileInput = $('#imageUpload')[0];
        
        if (fileInput.files.length === 0) {
            alert('请选择一张图像');
            return;
        }
        
        // 显示加载状态
        $('#loading').show();
        $('#resultContainer').hide();
        
        // 发送请求
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 隐藏加载状态
                $('#loading').hide();
                
                if (!response.success) {
                    alert('错误: ' + response.error);
                    return;
                }
                
                // 显示预测结果
                var realProb = (response.real_probability * 100).toFixed(2);
                var fakeProb = (response.fake_probability * 100).toFixed(2);
                
                $('#realProb').text(realProb + '%');
                $('#fakeProb').text(fakeProb + '%');
                
                $('#realProbBar').css('width', realProb + '%');
                $('#fakeProbBar').css('width', fakeProb + '%');
                
                if (response.prediction === 'REAL') {
                    $('#prediction').text('真实图像').removeClass('fake').addClass('real');
                } else {
                    $('#prediction').text('伪造图像').removeClass('real').addClass('fake');
                }
                
                // 显示结果图像
                $('#resultImage').attr('src', response.result_image);
                
                // 显示结果容器
                $('#resultContainer').fadeIn();
            },
            error: function() {
                $('#loading').hide();
                alert('请求处理过程中发生错误');
            }
        });
    });
    
    // 图像预览
    $('#imageUpload').on('change', function() {
        var fileInput = this;
        if (fileInput.files && fileInput.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                // 预览图像
                // 这里可以添加预览逻辑
            };
            reader.readAsDataURL(fileInput.files[0]);
        }
    });
});