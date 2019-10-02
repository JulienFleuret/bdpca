function [Y, X_, Wrt, Wct] = bdpca(X, krows, kcols)
% [Y, X, Wrt, Wct] = bdpca(X, krows, kcols)
%
% This function implement the Bi-Directional Principal Component Analysis
% detail in :
% 
% @inproceedings{zuo2005bi,
%   title={Bi-directional PCA with assembled matrix distance metric},
%   author={Zuo, Wangmeng and Wang, Kuanquan and Zhang, David},
%   booktitle={IEEE International Conference on Image Processing 2005},
%   volume={2},
%   pages={II--958},
%   year={2005},
%   organization={IEEE}
% }

%Copyright 2017 Julien FLEURET University Laval CVSL-MIVIM
%
%Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
%
%1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
%2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
%3. Neither the name of the copyright holder nor the names of its contributors
% may be used to endorse or promote products derived from this software without
% specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
% BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
% OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
% STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
% WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
% DAMAGE.


  if(nargin==1)
    krows=9;
    kcols = krows;
  end
  
  if(nargin==2)
    kcols = krows;  
  end
  
  if(~isa(X,'single') || ~isa(X,'double'))
    X = single(X);
  end
  
  % X is a batch of image.
  
  [rows, cols, frames] = size(X);
  
  %%
  % process the mean image.
  X_ = zeros(rows, cols);
  
  for i=1:frames
    
    X_ = X_ + X(:,:,i);
    
  end
  
  X_ = X_ / frames;
  
  %%
  % process the Scatter matrices.
  
  Srt = zeros(cols, cols);
  Sct = zeros(rows, rows);
  
  for i=1:frames
  
  A = X(:,:,i) - X_;
  
  Ar = A'*A;
  Ac = A*A';
  
  Srt = Srt + Ar;
  Sct = Sct + Ac;    
  
  end
  
  Srt = Srt / (frames * rows);
  Sct = Sct / (frames * cols);

  %%
  % process the eigenvectors of the Scatter matrices and keep on the krows, kcols
  % largest eigenvalues.  
  [Wrt,~] = eig(Srt);
  [Wct,~] = eig(Sct);

  Wrt = sort(Wrt,'descend');
  Wct = sort(Wct,'descend');

  
  Wrt = Wrt(:,1:krows);  
  Wct = Wct(:,1:kcols);
    
  Y = zeros(krows, kcols, frames);
  
  %%
  % process the features for every image of the batch.
  
  parfor i=1:frames
    
    Y(:,:,i) = Wct' * X(:,:,i) * Wrt;    
  
  end
  
  
  
end
