clear;clc;
file_path_simu='D:\add_brain_tex_try\sample_for_add_brain_tex\train_new_ran_17(3300)\';
file_all=dir([file_path_simu,'*.charles']); 
file_path_T2='D:\img_T2\';
T2_all=dir([file_path_T2,'*.mat']);
%%
close all;clc;
%chosen_num=18;
%for i=chosen_num:chosen_num
for i=2001:3300
    %%
    file_name=[file_path_simu,'\',file_all(i).name];
    geo_sap=Charles_reader(file_name,128,128);
    new_Re=permute(geo_sap,[2,3,1]);

    %figure;imshow3(new_Re(:,:,1:10),[],[2,5]);colormap gray;
    
    D_ori=new_Re(:,:,11);
    D=D_ori./max(max(D_ori));
    %figure;histogram(D);title('D geo hist');
    
    Dstar_ori=new_Re(:,:,12);
    Dstar=Dstar_ori./max(max(Dstar_ori));
    %figure;histogram(Dstar);title('Dstar geo hist');

    f_ori=new_Re(:,:,13);
    f=f_ori./max(max(f_ori));
    %figure;hist(f_ori);title('f geo hist'); 
    %%
    load([file_path_T2,'\',T2_all(i).name]);
    T2=imresize(T2_new,[128,128]);
    T2=T2./max(max(T2));
    
    mask=T2;
    mask=mask./max(mask(:));
    level=graythresh(mask);
    mask=im2bw(mask,level);
    T2_final=T2.*mask;
    %figure;hist(T2_final);title('ori T2 hist');
    
    match_T22D=(imhistmatch(T2_final, D)).*mask;

    match_T22Dstar=(imhistmatch(T2_final, Dstar)).*mask;

    match_T22f=(imhistmatch(T2_final, f)).*mask;

    new_D=match_T22D.*max(max(D_ori)); %figure;hist(match_T22D);title('match T22D hist')
    %figure;imshow(new_D,[]);colormap jet;
    new_Dstar=match_T22Dstar.*max(max(Dstar_ori));%figure;hist(match_T22Dstar);title('match T22Dstar hist');
    %figure;imshow(new_Dstar,[]);colormap jet;
    new_f=match_T22f.*max(max(f_ori)); %figure;hist(match_T22f);title('match T22f hist');
    %figure;imshow(new_f,[]);colormap jet;
    simu_b0=T2_final;

    slice=10;
    b_group = [0,50,100,150,200,300,400,600,800,1000];

    for j=1:slice
        simu_DWI(:,:,j)=(new_f.*exp(-b_group(j).*new_Dstar*10^-2)+(1-new_f).*exp(-b_group(j).*new_D*10^-3)).*simu_b0;
    end
    %figure;imshow3(simu_DWI,[0,1],[2,5]);colormap gray;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% add noise %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    signal=simu_b0(simu_b0~=0);
    snr_ratio=rand()*0.5+3;
    %snr_ratio=3.5;
    noise_value=mean(signal(:))/10^snr_ratio;
    
    for m=1:slice
        noise_real(:,:,:,m)=normrnd(0,noise_value,128,128).*mask;
        noise_im(:,:,:,m)=normrnd(0,noise_value,128,128).*mask;
    end
    
    im=zeros(128,128,slice);
    im_add_noise=zeros(128,128,slice);
    
    for k=1:slice
        simu_DWI(:,:,k)=simu_DWI(:,:,k)+noise_real(:,:,:,k);
        im_add_noise(:,:,k)=im(:,:,k)+noise_im(:,:,:,k);
        simu_DWI(:,:,k)=simu_DWI(:,:,k)+1i*im_add_noise(:,:,k);
        simu_DWI(:,:,k)=abs(simu_DWI(:,:,k));
    end
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 归一化 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    simu_DWI_norm=zeros(128,128,slice);
    test=simu_DWI(:,:,1)>0.1;
    for loop = 1:slice
        simu_DWI_norm(:,:,loop) = (simu_DWI(:,:,loop)./simu_DWI(:,:,1)).*test;
    end
    simu_DWI_norm(isnan(simu_DWI_norm))=0;
    simu_DWI_norm(isinf(simu_DWI_norm))=0;
    %figure;imshow3(simu_DWI_norm,[0,1]);colormap jet;
%%
    acco=max(max(simu_DWI_norm(:,:,10)));
    if acco<1
        %figure;imshow3(simu_DWI_norm,[0,1],[2,5]);colormap gray;
        brain_tex_test=cat(3,simu_DWI_norm,new_D,new_Dstar,new_f);
        brain_tex_test=permute(brain_tex_test,[3,1,2]);
        filepath='D:\2023_09_New_Project_IVIM_recon\2024.06.24_MP回复信实验\synthetic data with higher SNRs\17(3300)_30~35dB\';
        if exist(filepath)==0
            mkdir(filepath);
        end
        filenames=[filepath,'simu_brain_tex_30~35dB_',num2str(i),'.charles '];
        [fid,msg]=fopen(filenames,'wb');
        fwrite(fid,brain_tex_test,'double');
        fclose(fid);
        disp(['samp_num:',num2str(i)]);
    end
end

disp('completed!');
%%
% figure;
% subplot(221);imagesc(new_D,[0,5]);colormap jet;colorbar;
% subplot(222);imagesc(new_Dstar,[0,5]);colormap jet;colorbar;
% subplot(223);imagesc(new_f,[0,0.5]);colormap jet;colorbar;